//! Internal execution receipt for bounded graph-selection imputation.
//!
//! This module does not authorize a result. The caller must supply records produced by the
//! retained native program and independently replay them before issuing any claim.
use antecedent_core::CanonicalWriter;
use std::collections::BTreeSet;

const TAG: &str = "antecedent.graph_imputation.composite_receipt.v1";

/// Distinguishes scientifically unidentified mass from failed computation and a deliberate skip.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AtomStatus {
    Identified,
    Unidentified,
    #[allow(dead_code, reason = "failure receipts are reserved for a later licensed execution")]
    Failed,
    #[allow(dead_code, reason = "subsampling is not licensed in the bounded candidate")]
    Subsampled,
}

impl AtomStatus {
    fn tag(self) -> u64 {
        match self {
            Self::Identified => 0,
            Self::Unidentified => 1,
            Self::Failed => 2,
            Self::Subsampled => 3,
        }
    }
}

/// One graph in a single completed table's actual graph mixture.
#[derive(Clone, Debug)]
pub(crate) struct AtomReceipt {
    pub(crate) graph_mask: u64,
    /// Identity of the precompiled graph-specific operation; zero for an unidentified graph.
    pub(crate) program: Option<[u8; 32]>,
    /// Digest of the atom's retained identification result or explicit non-identification.
    pub(crate) identification: [u8; 32],
    pub(crate) weight: f64,
    pub(crate) status: AtomStatus,
    /// Digest of the native child execution, when one was actually attempted.
    pub(crate) execution: Option<[u8; 32]>,
}

/// Ordered receipt of one completed table. No mass is collapsed into another category.
#[derive(Clone, Debug)]
pub(crate) struct TableReceipt {
    pub(crate) index: u32,
    pub(crate) completed_input: [u8; 32],
    pub(crate) atoms: Vec<AtomReceipt>,
}

/// Digest and category masses of a sealed composite execution.
#[derive(Clone, Debug)]
pub(crate) struct CompositeReceipt {
    pub(crate) identity: [u8; 32],
    /// Per-table (unidentified, failed, skipped) weight; order is imputation order.
    pub(crate) masses: Vec<[f64; 3]>,
}

fn optional_digest(writer: &mut CanonicalWriter, value: Option<[u8; 32]>) {
    match value {
        Some(digest) => {
            writer.word(1).bytes(&digest);
        }
        None => {
            writer.word(0);
        }
    }
}

/// Seal only actual runtime receipts under a separately held program and incomplete-input identity.
/// This routine does not construct a coverage claim or decide scientific identification.
pub(crate) fn seal(
    program: [u8; 32],
    incomplete_input: [u8; 32],
    expected_tables: u32,
    tables: &[TableReceipt],
) -> Result<CompositeReceipt, &'static str> {
    if expected_tables < 2 || tables.len() != expected_tables as usize {
        return Err("wrong number of completed tables");
    }
    let mut writer = CanonicalWriter::new(TAG);
    writer.bytes(&program).bytes(&incomplete_input).word(u64::from(expected_tables));
    let mut masses = Vec::with_capacity(tables.len());
    for (index, table) in tables.iter().enumerate() {
        if table.index as usize != index || table.atoms.is_empty() {
            return Err("completed tables are missing or reordered");
        }
        writer.word(u64::from(table.index)).bytes(&table.completed_input);
        writer.word(table.atoms.len() as u64);
        let mut seen = BTreeSet::new();
        let mut total = 0.0;
        let mut lost = [0.0; 3];
        for atom in &table.atoms {
            if !seen.insert(atom.graph_mask)
                || !atom.weight.is_finite()
                || !(0.0..=1.0).contains(&atom.weight)
            {
                return Err("duplicate graph or invalid graph weight");
            }
            match atom.status {
                AtomStatus::Identified if atom.program.is_none() || atom.execution.is_none() => {
                    return Err("identified graph lacks its program or execution");
                }
                AtomStatus::Unidentified if atom.program.is_some() || atom.execution.is_some() => {
                    return Err("unidentified graph carries an execution");
                }
                AtomStatus::Failed if atom.program.is_none() || atom.execution.is_none() => {
                    return Err("failed graph lacks its prepared program or failure receipt");
                }
                AtomStatus::Subsampled if atom.execution.is_some() => {
                    return Err("subsampled graph carries an execution");
                }
                _ => {}
            }
            total += atom.weight;
            match atom.status {
                AtomStatus::Unidentified => lost[0] += atom.weight,
                AtomStatus::Failed => lost[1] += atom.weight,
                AtomStatus::Subsampled => lost[2] += atom.weight,
                AtomStatus::Identified => {}
            }
            writer
                .word(atom.graph_mask)
                .bytes(&atom.identification)
                .word(atom.weight.to_bits())
                .word(atom.status.tag());
            optional_digest(&mut writer, atom.program);
            optional_digest(&mut writer, atom.execution);
        }
        if (total - 1.0).abs() > 1e-12 {
            return Err("graph weights do not sum to one");
        }
        masses.push(lost);
    }
    Ok(CompositeReceipt { identity: writer.finish(), masses })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(index: u32) -> TableReceipt {
        TableReceipt {
            index,
            completed_input: [u8::try_from(index).unwrap(); 32],
            atoms: vec![
                AtomReceipt {
                    graph_mask: 1,
                    program: Some([1; 32]),
                    identification: [4; 32],
                    weight: 0.7,
                    status: AtomStatus::Identified,
                    execution: Some([2; 32]),
                },
                AtomReceipt {
                    graph_mask: 2,
                    program: None,
                    identification: [5; 32],
                    weight: 0.2,
                    status: AtomStatus::Unidentified,
                    execution: None,
                },
                AtomReceipt {
                    graph_mask: 3,
                    program: Some([3; 32]),
                    identification: [6; 32],
                    weight: 0.1,
                    status: AtomStatus::Failed,
                    execution: Some([7; 32]),
                },
            ],
        }
    }

    #[test]
    fn per_table_mass_is_distinct_and_all_source_mutations_change_identity() {
        let tables = vec![table(0), table(1)];
        let first = seal([8; 32], [9; 32], 2, &tables).unwrap();
        assert_eq!(first.masses, vec![[0.2, 0.1, 0.0]; 2]);
        assert_ne!(first.identity, seal([7; 32], [9; 32], 2, &tables).unwrap().identity);
        assert_ne!(first.identity, seal([8; 32], [7; 32], 2, &tables).unwrap().identity);
        let mut changed = tables.clone();
        changed[1].completed_input[0] ^= 1;
        assert_ne!(first.identity, seal([8; 32], [9; 32], 2, &changed).unwrap().identity);
        let mut changed = tables.clone();
        changed[1].atoms[0].weight = 0.6;
        changed[1].atoms[1].weight = 0.3;
        assert_ne!(first.identity, seal([8; 32], [9; 32], 2, &changed).unwrap().identity);
        let mut changed = tables.clone();
        changed[0].atoms[1].identification[0] ^= 1;
        assert_ne!(first.identity, seal([8; 32], [9; 32], 2, &changed).unwrap().identity);
        let mut changed = tables.clone();
        changed[0].atoms[0].execution = Some([6; 32]);
        assert_ne!(first.identity, seal([8; 32], [9; 32], 2, &changed).unwrap().identity);
    }

    #[test]
    fn invalid_and_unverifiable_receipts_refuse() {
        let tables = vec![table(0), table(1)];
        assert!(seal([8; 32], [9; 32], 3, &tables).is_err());
        let mut changed = tables.clone();
        changed.swap(0, 1);
        assert!(seal([8; 32], [9; 32], 2, &changed).is_err());
        let mut changed = tables.clone();
        changed[0].atoms[1].graph_mask = 1;
        assert!(seal([8; 32], [9; 32], 2, &changed).is_err());
        let mut changed = tables.clone();
        changed[0].atoms[0].weight = f64::NAN;
        assert!(seal([8; 32], [9; 32], 2, &changed).is_err());
        let mut changed = tables.clone();
        changed[0].atoms[0].execution = None;
        assert!(seal([8; 32], [9; 32], 2, &changed).is_err());
        let mut changed = tables;
        changed[0].atoms[1].status = AtomStatus::Subsampled;
        changed[0].atoms[1].execution = Some([2; 32]);
        assert!(seal([8; 32], [9; 32], 2, &changed).is_err());
    }
}
