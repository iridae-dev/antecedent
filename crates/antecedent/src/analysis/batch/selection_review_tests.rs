use super::*;

#[test]
fn embedded_influence_keeps_the_claims_own_variance() {
    // Six complete cases out of ten table rows. The claim's own variance is
    // S / (n (n - 1)) with S the centred sum of squares; the joint covariance of the
    // embedded column must reproduce it exactly.
    let inf = [1.0, -2.0, 0.5, 3.0, -1.5, 4.0];
    let rows = [0usize, 2, 3, 5, 6, 9];
    let n = inf.len() as f64;
    let mean = inf.iter().sum::<f64>() / n;
    let s: f64 = inf.iter().map(|v| (v - mean).powi(2)).sum();
    let own_variance = s / (n * (n - 1.0));
    let col = embed_centered_influence(&inf, &rows, 10);
    let cov = antecedent_estimate::joint_influence_covariance(&[col.as_slice()], None).unwrap();
    assert!(
        (cov.get(0, 0) - own_variance).abs() < 1e-12 * own_variance,
        "joint variance {} vs own {own_variance}",
        cov.get(0, 0)
    );
}

#[test]
fn recorded_splits_reject_empty_duplicate_and_out_of_range_rows() {
    let values = [0.0, 1.0, 2.0];
    let data = TabularData::from_f64_columns([("x", values.as_slice())]).unwrap();
    for (screen_rows, estimate_rows) in [
        (vec![], vec![1]),
        (vec![0], vec![]),
        (vec![0, 0], vec![1]),
        (vec![3], vec![1]),
        (vec![0], vec![1, 1]),
        (vec![0], vec![3]),
    ] {
        let screen = CandidateScreen {
            screen_id: "review".into(),
            procedure: CandidateProcedure::MaxT,
            screen_rows: screen_rows
                .into_iter()
                .map(|id| antecedent_data::GlobalId::new(u64::try_from(id).unwrap()))
                .collect::<Vec<_>>()
                .into(),
            estimate_rows: estimate_rows
                .into_iter()
                .map(|id| antecedent_data::GlobalId::new(u64::try_from(id).unwrap()))
                .collect::<Vec<_>>()
                .into(),
        };
        assert!(subset_estimate_rows(&data, Some(&screen)).is_err());
    }
}
