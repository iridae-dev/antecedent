use antecedent_core::{AverageEffectQuery, CausalQuery, ExecutionContext, VariableId};
use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, LaplaceWorkspace, PriorSet, PriorSpec, fit_conjugate_gaussian,
};
use antecedent_stats::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependenceTest, ConfidenceMethod,
    KnnDependence, SignificanceMethod,
};

fn design<'a>(x: &'a [f64], y: &'a [f64]) -> BayesDesignRef<'a> {
    BayesDesignRef { x_colmajor: x, y, nrows: 32, ncols: 1, weights: None, offsets: None }
}

fn main() {
    let mut dag = antecedent_graph::Dag::with_variables(4);
    // T=0, Y=1, Z=2, U=3: a treatment descendant is not an exogenous IV.
    for (a, b) in [(0, 1), (0, 2), (3, 0), (3, 1)] {
        dag.insert_directed(
            antecedent_graph::DenseNodeId::from_raw(a),
            antecedent_graph::DenseNodeId::from_raw(b),
        )
        .unwrap();
    }
    let iv = antecedent_identify::InstrumentalVariableIdentifier::new();
    let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    ));
    let result = iv
        .identify(
            &iv.prepare(&dag).unwrap(),
            &query,
            &mut antecedent_identify::IdentificationWorkspace::default(),
        )
        .unwrap();
    println!(
        "IV descendant: status={:?}, instruments={:?}",
        result.status,
        result.estimands.iter().map(|e| &e.instruments).collect::<Vec<_>>()
    );
    {
        use antecedent_estimate::OverlapPolicy;
        use antecedent_estimate::iv::{PreparedIvProblem, TwoStageLeastSquares, WaldIv};
        use std::sync::Arc;
        let z = [0., 0., 0., 0., 1., 1., 1., 1.];
        let u = [0., 0., 0., 1., 0., 1., 1., 1.];
        let t: Vec<f64> = z.iter().zip(u).map(|(z, u)| z + u).collect();
        let y: Vec<f64> = t.iter().zip(u).map(|(t, u)| 2. * t + 3. * u).collect();
        let mut zm = vec![1.; 8];
        zm.extend(z);
        let mut xm = vec![1.; 8];
        xm.extend(u);
        let p = PreparedIvProblem {
            instruments_matrix: zm.into(),
            z_ncols: 2,
            exogenous_matrix: xm.into(),
            x_ncols: 2,
            nrows: 8,
            treatment: t.into(),
            outcome: y.into(),
            method: Arc::from("iv"),
            instruments: Arc::from([VariableId::from_raw(2)]),
            adjustment_set: Arc::from([VariableId::from_raw(3)]),
            overlap: OverlapPolicy::ExplicitOverride,
            treatment_delta: 1.,
        };
        let ctx = ExecutionContext::for_tests(42);
        let wald =
            WaldIv::new().with_bootstrap_replicates(0).fit(&p, &ctx, Default::default()).unwrap();
        let tsls = TwoStageLeastSquares::new()
            .with_bootstrap_replicates(0)
            .fit(&p, &mut Default::default(), &ctx, Default::default())
            .unwrap();
        println!("conditional IV: wald={}, 2sls={}, truth=2", wald.ate, tsls.ate);
    }
    let options = BayesFitOptions { n_draws: 10, ..Default::default() };
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::KnownResidualVariance(1.0));
    let mut ws = LaplaceWorkspace::default();
    let y = vec![2.0; 32];
    let x = vec![1.0; 32];
    let ptr = x.as_ptr();
    let initial = fit_conjugate_gaussian(design(&x, &y), &prior, &options, &mut ws).unwrap();
    drop(initial);
    drop(x);
    let mut recycled = false;
    let mut allocations = Vec::new();
    for _ in 0..10000 {
        let x = vec![2.0; 32];
        if x.as_ptr() == ptr {
            let reused = fit_conjugate_gaussian(design(&x, &y), &prior, &options, &mut ws).unwrap();
            let fresh = fit_conjugate_gaussian(
                design(&x, &y),
                &prior,
                &options,
                &mut LaplaceWorkspace::default(),
            )
            .unwrap();
            println!(
                "conjugate recycled allocation: reused={}, fresh={}",
                reused.map[0], fresh.map[0]
            );
            recycled = true;
            break;
        }
        allocations.push(x);
    }
    println!("conjugate allocator recycle observed={recycled}");

    let ctx = ExecutionContext::for_tests(42);
    let test = KnnDependence::new(2);
    let mut ws = CiWorkspace::default();
    let x: Vec<_> = (0..30).map(|i| i as f64).collect();
    let mut y = x.clone();
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
    let eval = |y: &[f64], ws: &mut CiWorkspace| {
        let columns = [&x[..], y];
        let req = CiBatchRequest {
            columns: &columns,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::BlockShuffle { replicates: 9, block_size: 1 },
            confidence: ConfidenceMethod::None,
        };
        test.test_batch_adhoc(&req, ws, &ctx).unwrap().results[0]
    };
    let before = eval(&y, &mut ws);
    y[7] = 1000.0;
    let reused = eval(&y, &mut ws);
    let fresh = eval(&y, &mut CiWorkspace::default());
    println!("knn: before={before:?}, reused={reused:?}, fresh={fresh:?}");

    for m in [10, 16, 20] {
        let mut dag = antecedent_graph::Dag::with_variables(2 + m);
        dag.insert_directed(
            antecedent_graph::DenseNodeId::from_raw(0),
            antecedent_graph::DenseNodeId::from_raw(1),
        )
        .unwrap();
        let id = antecedent_identify::BackdoorIdentifier::new();
        let prep = id.prepare(&dag).unwrap();
        let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let start = std::time::Instant::now();
        let result = id
            .identify(&prep, &query, &mut antecedent_identify::IdentificationWorkspace::default())
            .unwrap();
        println!(
            "backdoor irrelevant={m}: performance={:?}, elapsed={:?}",
            result.performance,
            start.elapsed()
        );
    }
    let mut stats = antecedent_state::LinearOlsSuffStats::new(1);
    for e in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        stats.append_row(&[1.0], 1e12 + e).unwrap();
    }
    let beta = stats.solve_beta().unwrap();
    println!("streaming OLS offset variance={:?}; expected=2.5", stats.residual_variance(&beta));
    use antecedent_state::{GraphScoreData, GraphScoreFamily, full_graph_score};
    use std::{collections::HashMap, sync::Arc};
    let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.31).sin()).collect();
    for scale in [1.0, 1e-8] {
        let mut columns = x.clone();
        columns.extend(
            x.iter().enumerate().map(|(i, x)| scale * (2.0 * x + 0.1 * (i as f64 * 1.71).cos())),
        );
        let data = GraphScoreData::new(100, 2, Arc::from(columns)).unwrap();
        let empty =
            full_graph_score(&data, GraphScoreFamily::GaussianBic, &HashMap::new()).unwrap();
        let mut parents = HashMap::new();
        parents.insert(1, Arc::from([0u32]));
        let edge = full_graph_score(&data, GraphScoreFamily::GaussianBic, &parents).unwrap();
        println!("BIC outcome scale={scale}: edge-minus-empty={}", edge - empty);
    }
}
