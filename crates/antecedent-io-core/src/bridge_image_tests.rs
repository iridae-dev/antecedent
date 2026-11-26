// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::bridge_image_artifact::{BridgeImageRequest, StoredBridgeImage};
use crate::bridge_joint_outcomes::{JointOutcomeMap, JointOutcomeTerm};
use antecedent_contract::bridge::{FixedBridgeDomain, IdentifiedSetSource, ValueInterval};
use antecedent_core::IdentifiedSet;

#[test]
fn fixed_bridge_maps_sharp_sets_including_decreasing_and_constant_images() {
    let source = IdentifiedSetSource {
        quantity: quantity("trial:biomarker", "mmol_per_l"),
        set: IdentifiedSet::try_new(-2.0, 3.0).unwrap(),
        claim_identity: SemanticDigest::from_bytes([31; 32]),
    };
    let domain = FixedBridgeDomain(ValueInterval { low: -2., high: 4. });
    for (slope, lower, upper) in [(2., -3., 7.), (-2., -5., 5.), (0., 1., 1.)] {
        let map = contract(
            BridgeMapping::Affine { intercept: 1.0, slope },
            ValueKind::IdentifiedSet,
            None,
            None,
        );
        let image = map.identified_set_image(&source, domain).unwrap();
        assert_eq!(image.set, IdentifiedSet::try_new(lower, upper).unwrap());
        assert!(
            map.identified_set_image(
                &source,
                FixedBridgeDomain(ValueInterval { low: -1., high: 4. })
            )
            .is_err()
        );
        verify_portable_request(BridgeImageRequest::IdentifiedSet {
            bridge: Box::new(map.clone()),
            source: source.clone(),
            domain,
        });
        let mut changed = source.clone();
        changed.claim_identity = SemanticDigest::from_bytes([13; 32]);
        assert_ne!(map.identified_set_image(&changed, domain).unwrap().identity, image.identity);
    }
}

#[test]
fn affine_vector_map_preserves_the_weighted_joint_table_and_refuses_unpaired_inputs() {
    let input = table(
        &[1., 2., 4.],
        &[3., 2., 1.],
        DistributionMeaningWire::EmpiricalOutcome,
        DrawAlignment::Joint,
    );
    let mut metadata = input.metadata().clone();
    metadata.weights = Some(vec![2., 1., 3.]);
    let source = DistributionArtifact::new(metadata.clone(), input.draws().to_vec()).unwrap();
    let terms = |input: usize, output: &str, offset: f64, slope: f64| {
        let mut declaration = contract(
            BridgeMapping::Affine { intercept: offset, slope },
            ValueKind::JointOutcomeLaw,
            None,
            None,
        );
        declaration.source.quantity =
            ScientificQuantity::try_from(source.quantities()[input].clone()).unwrap();
        declaration.target.quantity = quantity(output, "log_hazard");
        JointOutcomeTerm {
            contract: declaration,
            domain: FixedBridgeDomain(ValueInterval { low: 0., high: 5. }),
        }
    };
    let map = JointOutcomeMap {
        outputs: vec![
            vec![terms(1, "total", 0., 2.), terms(0, "total", 1., 1.)],
            vec![terms(0, "difference", 0., 1.), terms(1, "difference", 0., -1.)],
        ],
    };
    let ctx = ExecutionContext::for_tests(91);
    let image = map.apply(&source, &ctx).unwrap();
    assert_eq!(image.distribution.draws(), &[8., -2., 7., 0., 7., 3.]);
    assert_eq!(image.distribution.metadata().weights, metadata.weights);
    assert_eq!(
        image.distribution.alignment_identity().unwrap(),
        source.alignment_identity().unwrap()
    );
    near(image.distribution.mean(1).unwrap(), 5. / 6., 1e-12);
    verify_portable_request(BridgeImageRequest::JointOutcomes {
        map: map.clone(),
        source: source.clone(),
    });
    metadata.identity.alignment = DrawAlignment::IndependentMarginals;
    let unpaired = DistributionArtifact::new(metadata, input.draws().to_vec()).unwrap();
    assert!(map.apply(&unpaired, &ctx).is_err());
    assert!(map.apply(&source, &ctx.clone().with_memory_limit(Some(1))).is_err());
    let mut changed = map.clone();
    changed.outputs[0][0].domain.0.high = 2.;
    assert!(changed.apply(&source, &ctx).is_err());
    changed.outputs[0][0] = terms(1, "total", 0., 2.);
    changed.outputs[0][0].contract.source.quantity.units = "another_scale".into();
    assert!(changed.apply(&source, &ctx).is_err());
}

fn verify_portable_request(request: BridgeImageRequest) {
    let ctx = ExecutionContext::for_tests(13);
    let result = request.evaluate(&ctx).unwrap();
    let program = SemanticDigest::from_bytes([11; 32]);
    let bytes = request.export(&result, program, "exact-image").unwrap();
    let stored = StoredBridgeImage::open(&bytes, program, result.identity()).unwrap();
    let computed = stored.request.evaluate(&ctx).unwrap();
    stored.verify(program, &computed).unwrap();
    let other = SemanticDigest::from_bytes([2; 32]);
    assert!(StoredBridgeImage::open(&bytes, other, result.identity()).is_err());
    assert!(StoredBridgeImage::open(&bytes, program, other).is_err());
    assert!(stored.verify(other, &computed).is_err());
}
