//! The population do-contrast attribution target: its encoding, its validation and the variable
//! it is about.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::query::{ModelOperation, ModelOperationFault};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn digest<T: CanonicalEncode>(value: &T) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("t");
    value.encode(&mut writer);
    writer.finish()
}

fn contrast() -> PopulationDoContrastTarget {
    PopulationDoContrastTarget { parent: v(0), child: v(2), delta: 0.5, max_units: None, seed: 41 }
}

#[test]
fn the_contrast_target_is_tagged_apart_from_its_operation_and_reads_every_field() {
    let target = contrast();
    let query = AttributeQuery::new(AttributeTarget::PopulationDoContrast(target));
    let mut expected = CanonicalWriter::new("t");
    expected.word(10);
    target.encode(&mut expected);
    assert_eq!(digest(&query), expected.finish());
    assert_ne!(digest(&query), digest(&ModelOperation::PopulationDoContrast(target)));
    assert_eq!(query.target.as_str(), "population_do_contrast");
    let edits: [PopulationDoContrastTarget; 5] = [
        PopulationDoContrastTarget { parent: v(1), ..target },
        PopulationDoContrastTarget { child: v(3), ..target },
        PopulationDoContrastTarget { delta: 1.0, ..target },
        PopulationDoContrastTarget { max_units: Some(10), ..target },
        PopulationDoContrastTarget { seed: 42, ..target },
    ];
    for edited in edits {
        let other = AttributeQuery::new(AttributeTarget::PopulationDoContrast(edited));
        assert_ne!(digest(&other), digest(&query));
    }
}

#[test]
fn the_contrast_query_refuses_a_null_width_and_a_self_contrast_and_is_about_the_child() {
    let ok = AttributeQuery::new(AttributeTarget::PopulationDoContrast(contrast()));
    assert!(ok.validate().is_ok());
    assert_eq!(ok.outcome(), Ok(v(2)));
    let null =
        AttributeQuery::new(AttributeTarget::PopulationDoContrast(PopulationDoContrastTarget {
            delta: 0.0,
            ..contrast()
        }));
    assert_eq!(
        null.validate(),
        Err(QueryError::InvalidModelOperation(ModelOperationFault::ContrastWidth))
    );
    let same =
        AttributeQuery::new(AttributeTarget::PopulationDoContrast(PopulationDoContrastTarget {
            child: v(0),
            ..contrast()
        }));
    assert_eq!(
        same.validate(),
        Err(QueryError::InvalidModelOperation(ModelOperationFault::ContrastSameVariable))
    );
}
