use super::*;

/// The shared assignment is a balanced shuffle keyed by the master seed: fold sizes
/// differ by at most one, the same seed reproduces it, another seed changes it, and
/// it is not the periodic `row % k` pattern that aliases with alternating designs.
#[test]
fn shared_folds_are_a_seeded_balanced_shuffle_not_row_modulo() {
    let n = 103usize;
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let data = TabularData::from_f64_columns([("x", values.as_slice())]).unwrap();
    let compile = |seed: u64| SharedBatchDesign::compile(&data, &[], 5, seed).unwrap();
    let a = compile(7);
    let mut counts = [0usize; 5];
    for &fold in a.fold_ids.iter() {
        counts[fold as usize] += 1;
    }
    // 103 rows dealt round-robin over 5 folds: sizes 21, 21, 21, 20, 20.
    counts.sort_unstable();
    assert_eq!(counts, [20, 20, 21, 21, 21]);
    assert_eq!(a.fold_ids, compile(7).fold_ids);
    assert_ne!(a.fold_ids, compile(8).fold_ids);
    let modulo: Vec<u32> = (0..n).map(|i| u32::try_from(i % 5).unwrap()).collect();
    assert_ne!(a.fold_ids.to_vec(), modulo);
    assert_eq!(a.fold_seed, 7);
    // Rebinding to a same-size table keeps the seed, hence the assignment.
    assert_eq!(a.rebind(&data).unwrap().fold_ids, a.fold_ids);
}
