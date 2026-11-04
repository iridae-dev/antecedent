//! Exact-request verification of an external Bernoulli(1/4) law, with the provider's answers
//! checked against the law enumerated by hand.
//!
//! The expected side is the enumerated Bernoulli(1/4): total mass 1, mean 1/4, CDF(0) = 3/4 and
//! largest support point 1 (all dyadic, so exact in binary floating point). The observed side comes
//! from a separate test-side provider that builds its law differently: by enumerating the four
//! equiprobable outcomes of two fair coins and reading `Y = coin_1 AND coin_2`, then answering mass,
//! mean, CDF and support queries from its own atoms. `verify_external_object` must issue a receipt
//! for the provider whose answers equal the enumeration, and must refuse the providers whose
//! answers differ from it, at the probe that sees the difference: an OR of the coins (p = 3/4), a
//! single coin (p = 1/2), an AND of three coins (p = 1/8), a law with the right mean but the
//! wrong CDF, and an unnormalized law. The tolerance 1e-12 only absorbs float noise, as every
//! quantity is a dyadic rational.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    DistributionMeaning, ExternalCapability, ExternalScientificObject, ExternalVerificationError,
    LawProviderContract, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
    VerificationProbe, VerificationProbeKind as P, verify_external_object,
};

const TOLERANCE: f64 = 1e-12;

/// A discrete law as the provider holds it: `(value, mass)` atoms.
struct Provider {
    atoms: Vec<(f64, f64)>,
}

impl Provider {
    /// Enumerate `k` fair coins and apply `rule` to the outcome: each of the `2^k` outcomes has
    /// mass `2^-k`, and the atoms are the values the rule takes.
    fn from_coins(k: u32, rule: impl Fn(&[bool]) -> bool) -> Self {
        let mut mass_one = 0.0;
        let mut mass_zero = 0.0;
        for outcome in 0..(1_u32 << k) {
            let coins: Vec<bool> = (0..k).map(|bit| (outcome >> bit) & 1 == 1).collect();
            let weight = 0.5_f64.powi(i32::try_from(k).expect("coin count fits i32"));
            if rule(&coins) {
                mass_one += weight;
            } else {
                mass_zero += weight;
            }
        }
        Self { atoms: vec![(0.0, mass_zero), (1.0, mass_one)] }
    }

    fn mass(&self) -> f64 {
        self.atoms.iter().map(|(_, m)| m).sum()
    }

    fn mean(&self) -> f64 {
        self.atoms.iter().map(|(v, m)| v * m).sum()
    }

    fn cdf(&self, x: f64) -> f64 {
        self.atoms.iter().filter(|(v, _)| *v <= x).map(|(_, m)| m).sum()
    }

    fn support_max(&self) -> f64 {
        self.atoms.iter().map(|(v, _)| *v).fold(f64::MIN, f64::max)
    }

    /// Seeded inverse-CDF draws from a splitmix64 stream.
    fn seeded_draws(&self, seed: u64, n: usize) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = (z >> 11) as f64 / (1_u64 << 53) as f64;
                let mut acc = 0.0;
                for (value, mass) in &self.atoms {
                    acc += mass;
                    if u < acc {
                        return *value;
                    }
                }
                self.support_max()
            })
            .collect()
    }

    /// The six probes the law's capabilities (sample, mean, CDF) require, observed from the
    /// provider and compared with the enumerated Bernoulli(1/4).
    fn probes(&self) -> Vec<VerificationProbe> {
        let draws = self.seeded_draws(3, 64);
        let agree = f64::from(u8::from(draws == self.seeded_draws(3, 64)));
        let in_support = f64::from(u8::from(draws.iter().all(|x| [0.0, 1.0].contains(x))));
        let make = |kind, observed, expected| VerificationProbe {
            kind,
            observed,
            expected,
            tolerance: TOLERANCE,
        };
        vec![
            make(P::Shape, 1.0, 1.0),
            make(P::Normalization, self.mass(), 1.0),
            make(P::Moments, self.mean(), 0.25),
            make(P::KnownTruth, self.cdf(0.0), 0.75),
            make(P::Support, self.support_max() * in_support, 1.0),
            make(P::SeededBehavior, agree, 1.0),
        ]
    }
}

fn object() -> ExternalScientificObject {
    ExternalScientificObject::Law(LawProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "bernoulli".into(),
            version_id: "v1".into(),
            snapshot_id: "snap-1".into(),
            request_id: "R1".into(),
        },
        quantities: vec![ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "y".into(),
            role: QuantityRole::Outcome,
            units: "units".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }],
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![
            ExternalCapability::Sample,
            ExternalCapability::Mean,
            ExternalCapability::Cdf,
        ],
    })
}

fn refused_at(provider: &Provider) -> P {
    match verify_external_object(&object(), &provider.probes()) {
        Err(ExternalVerificationError::FailedProbe(kind)) => kind,
        other => panic!("expected a failed probe, got {other:?}"),
    }
}

#[test]
fn a_provider_whose_answers_equal_the_enumerated_law_is_verified() {
    let provider = Provider::from_coins(2, |c| c[0] && c[1]);
    assert_eq!(provider.atoms, vec![(0.0, 0.75), (1.0, 0.25)]);
    let receipt = verify_external_object(&object(), &provider.probes()).unwrap();
    assert!(receipt.covers(&object()));
}

#[test]
fn providers_whose_answers_differ_from_the_enumerated_law_are_refused_at_the_probe_that_sees_it() {
    // p = 3/4 (OR of two coins): mean 3/4 instead of 1/4.
    assert_eq!(refused_at(&Provider::from_coins(2, |c| c[0] || c[1])), P::Moments);
    // p = 1/2 (a single coin).
    assert_eq!(refused_at(&Provider::from_coins(1, |c| c[0])), P::Moments);
    // p = 1/8 (AND of three coins).
    assert_eq!(refused_at(&Provider::from_coins(3, |c| c[0] && c[1] && c[2])), P::Moments);
    // Mean 1/4 but the wrong law: mass 1/2 at 0 and 1/2 at 1/2. Only the CDF sees it.
    let wrong_cdf = Provider { atoms: vec![(0.0, 0.5), (0.5, 0.5)] };
    assert!((wrong_cdf.mean() - 0.25).abs() < 1e-12);
    assert_eq!(refused_at(&wrong_cdf), P::KnownTruth);
    // Right CDF(0) but total mass 7/8: only Normalization sees it.
    let unnormalized = Provider { atoms: vec![(0.0, 0.75), (1.0, 0.125)] };
    assert_eq!(refused_at(&unnormalized), P::Normalization);
}
