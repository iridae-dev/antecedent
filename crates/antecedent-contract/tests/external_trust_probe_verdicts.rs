//! Verification verdicts of a nontrivial provider law, from hand-computed probe values.
//!
//! The declared law is a four-atom pmf on `x in {0, 1, 2, 3}`, `p = (0.1, 0.2, 0.3, 0.4)`.
//! Its five probe quantities are computed by enumeration of the pmf (never by the library),
//! and every candidate provider is itself a different pmf whose reported quantities are
//! hand-derived below. Each verdict is therefore read from a hand-written gap table against
//! the declared tolerance, not produced by repeating the comparison the library makes:
//!
//! | probe kind      | quantity                   | truth | tolerance |
//! |-----------------|----------------------------|-------|-----------|
//! | KnownTruth      | mean `E[x]`                | 2.0   | 0.125     |
//! | Shape           | `P(x <= 1)`                | 0.3   | 0.0625    |
//! | Support         | largest atom with mass     | 3     | 0.0       |
//! | Moments         | variance `E[x^2] - E[x]^2` | 1.0   | 0.25      |
//! | SeededBehavior  | top-atom mass `P(x = 3)`   | 0.4   | 0.05      |
//!
//! Hand derivations (`E[x^2] = p1 + 4 p2 + 9 p3`):
//!
//! * truth: `E[x] = 0.2 + 0.6 + 1.2 = 2`, `E[x^2] = 0.2 + 1.2 + 3.6 = 5`, variance `5 - 4 = 1`.
//! * B `(0.15, 0.15, 0.30, 0.40)`: mean 1.95 (gap 0.05), variance `4.95 - 3.8025 = 1.1475`
//!   (gap 0.1475): inside both tolerances.
//! * C `(0.1, 0.2, 0.4, 0.3)`: mean 1.9 (gap 0.1, inside), variance `4.5 - 3.61 = 0.89`
//!   (gap 0.11, inside), top-atom mass 0.3 (gap 0.1 > 0.05): outside only that probe.
//! * D `(0, 0.3, 0.3, 0.4)`: mean 2.1 (gap 0.1, inside), `P(x <= 1) = 0.3`, variance
//!   `5.1 - 4.41 = 0.69` (gap 0.31 > 0.25): outside only the moment probe.
//! * E `(1/6, 1/3, 1/2, 0)` (support truncated at 2): mean 4/3 (gap 2/3), `P(x <= 1) = 1/2`
//!   (gap 0.2), largest atom 2 (gap 1), variance `7/3 - 16/9 = 5/9` (gap 4/9), top-atom
//!   mass 0 (gap 0.4): outside every probe.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    DistributionMeaning, ExternalCapability, ExternalScientificObject, ExternalVerificationError,
    LawProviderContract, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
    VerificationProbe, VerificationProbeKind, verify_external_object,
};

/// Probe kinds in the order the library walks them (the order of the hand table).
const KINDS: [VerificationProbeKind; 5] = [
    VerificationProbeKind::KnownTruth,
    VerificationProbeKind::Shape,
    VerificationProbeKind::Support,
    VerificationProbeKind::Moments,
    VerificationProbeKind::SeededBehavior,
];

/// Declared tolerance of each probe, in `KINDS` order.
const TOLERANCE: [f64; 5] = [0.125, 0.0625, 0.0, 0.25, 0.05];

/// The declared law.
const TRUE_LAW: [f64; 4] = [0.1, 0.2, 0.3, 0.4];

/// The five probe quantities of a pmf on `{0, 1, 2, 3}`, by enumeration.
fn quantities(pmf: &[f64; 4]) -> [f64; 5] {
    let mut mean = 0.0;
    let mut second = 0.0;
    let mut cdf_at_one = 0.0;
    let mut top_atom = 0.0;
    for (x, &p) in pmf.iter().enumerate() {
        let x = x as f64;
        mean += x * p;
        second += x * x * p;
        if x <= 1.0 {
            cdf_at_one += p;
        }
        if p > 0.0 {
            top_atom = x;
        }
    }
    [mean, cdf_at_one, top_atom, second - mean * mean, pmf[3]]
}

struct Candidate {
    name: &'static str,
    pmf: [f64; 4],
    /// Hand-derived quantities (see the module table), in `KINDS` order.
    by_hand: [f64; 5],
    /// Hand-derived: is the gap to the truth strictly greater than the tolerance?
    outside: [bool; 5],
}

fn candidates() -> Vec<Candidate> {
    vec![
        Candidate {
            name: "A exact",
            pmf: [0.1, 0.2, 0.3, 0.4],
            by_hand: [2.0, 0.3, 3.0, 1.0, 0.4],
            outside: [false; 5],
        },
        Candidate {
            name: "B small perturbation",
            pmf: [0.15, 0.15, 0.30, 0.40],
            by_hand: [1.95, 0.3, 3.0, 1.1475, 0.4],
            outside: [false; 5],
        },
        Candidate {
            name: "C top atom too light",
            pmf: [0.1, 0.2, 0.4, 0.3],
            by_hand: [1.9, 0.3, 3.0, 0.89, 0.3],
            outside: [false, false, false, false, true],
        },
        Candidate {
            name: "D variance too small",
            pmf: [0.0, 0.3, 0.3, 0.4],
            by_hand: [2.1, 0.3, 3.0, 0.69, 0.4],
            outside: [false, false, false, true, false],
        },
        Candidate {
            name: "E support truncated",
            pmf: [1.0 / 6.0, 1.0 / 3.0, 0.5, 0.0],
            by_hand: [4.0 / 3.0, 0.5, 2.0, 5.0 / 9.0, 0.0],
            outside: [true; 5],
        },
    ]
}

fn quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "x".into(),
        variable_name: "x".into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn identity(version: &str) -> ProviderObjectIdentity {
    ProviderObjectIdentity {
        provider_id: "lab".into(),
        object_id: "law-x".into(),
        version_id: version.into(),
        snapshot_id: "snap-1".into(),
        request_id: "req-1".into(),
    }
}

fn object(version: &str) -> ExternalScientificObject {
    ExternalScientificObject::Law(LawProviderContract {
        identity: identity(version),
        quantities: vec![quantity()],
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![ExternalCapability::Sample, ExternalCapability::Mean],
    })
}

fn probes(truth: &[f64; 5], observed: &[f64; 5]) -> Vec<VerificationProbe> {
    (0..5)
        .map(|k| VerificationProbe {
            kind: KINDS[k],
            observed: observed[k],
            expected: truth[k],
            tolerance: TOLERANCE[k],
        })
        .collect()
}

/// The enumerated quantities equal the hand derivations in the module table.
#[test]
fn the_enumerated_probe_quantities_equal_the_hand_derivations() {
    let truth = quantities(&TRUE_LAW);
    for (got, want) in truth.iter().zip([2.0, 0.3, 3.0, 1.0, 0.4]) {
        assert!((got - want).abs() < 1e-12, "truth {got} vs hand {want}");
    }
    for candidate in candidates() {
        let computed = quantities(&candidate.pmf);
        for k in 0..5 {
            assert!(
                (computed[k] - candidate.by_hand[k]).abs() < 1e-12,
                "{}: probe {k} enumerates to {} but was derived by hand as {}",
                candidate.name,
                computed[k],
                candidate.by_hand[k]
            );
        }
    }
}

/// Each probe alone is the deviant (the other four report the truth): the verdict is the
/// hand-derived `outside` flag, for every provider and every probe kind.
#[test]
fn each_probe_verdict_follows_the_hand_derived_gap_against_its_tolerance() {
    let truth = quantities(&TRUE_LAW);
    let law = object("v1");
    for candidate in candidates() {
        for k in 0..5 {
            let mut observed = truth;
            observed[k] = candidate.by_hand[k];
            let verdict = verify_external_object(&law, &probes(&truth, &observed));
            if candidate.outside[k] {
                assert_eq!(
                    verdict.err(),
                    Some(ExternalVerificationError::FailedProbe(KINDS[k])),
                    "{}: probe {k} is outside its tolerance",
                    candidate.name
                );
            } else {
                assert!(verdict.is_ok(), "{}: probe {k} is inside its tolerance", candidate.name);
            }
        }
    }
}

/// All five probes together: the receipt is issued exactly for providers with no probe
/// outside tolerance, otherwise the first (in probe order) failing kind is named; an issued
/// receipt carries all five probes and covers the verified request and no neighbour.
#[test]
fn a_provider_receives_a_receipt_only_when_every_hand_derived_gap_is_within_tolerance() {
    let truth = quantities(&TRUE_LAW);
    let law = object("v1");
    for candidate in candidates() {
        let verdict = verify_external_object(&law, &probes(&truth, &candidate.by_hand));
        match candidate.outside.iter().position(|&outside| outside) {
            None => {
                let receipt = verdict.unwrap_or_else(|e| panic!("{}: {e:?}", candidate.name));
                assert!(receipt.covers(&law), "{}", candidate.name);
                assert_eq!(receipt.probes_for(&law).map(<[_]>::len), Some(5));
                assert!(!receipt.covers(&object("v2")), "{}: another version", candidate.name);
            }
            Some(first) => assert_eq!(
                verdict.err(),
                Some(ExternalVerificationError::FailedProbe(KINDS[first])),
                "{}",
                candidate.name
            ),
        }
    }
}
