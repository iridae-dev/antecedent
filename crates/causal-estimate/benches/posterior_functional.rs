//! Batched posterior functional evaluation benchmark .
#![allow(missing_docs, clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::ExecutionContext;
use causal_estimate::{CompiledGCompAte, GCompAteEvaluator, PosteriorFunctionalEvaluator};
use causal_prob::{EffectBatch, PosteriorDraws, PosteriorEvalWorkspace, PosteriorSchema};
use causal_stats::GlmFamily;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_gcomp_eval(c: &mut Criterion) {
    let nrows = 400usize;
    let ncols = 3usize;
    let n_draws = 512usize;
    let mut matrix = vec![0.0; nrows * ncols];
    for r in 0..nrows {
        matrix[r] = 1.0;
        matrix[nrows + r] = (r % 2) as f64;
        matrix[2 * nrows + r] = (r as f64) * 0.01;
    }
    let mut coef_vals = vec![0.0; n_draws * ncols];
    for d in 0..n_draws {
        coef_vals[d] = 0.1;
        coef_vals[n_draws + d] = 2.0;
        coef_vals[2 * n_draws + d] = 0.3;
    }
    let draws =
        PosteriorDraws::from_column_major(PosteriorSchema::coefficients(ncols), n_draws, coef_vals)
            .unwrap();
    let evaluator = GCompAteEvaluator {
        family: GlmFamily::GaussianIdentity,
        treatment_col: 1,
        active: 1.0,
        control: 0.0,
        nrows,
        ncols,
        matrix: Arc::from(matrix),
    };
    let compiled = evaluator.compile().unwrap();
    let mut ws = PosteriorEvalWorkspace::default();
    ws.prepare(n_draws, ncols);
    let grow0 = ws.grow_count;
    let mut out = EffectBatch::default();
    out.prepare(n_draws);
    let batch = draws.batch(0, n_draws).unwrap();
    let ctx = ExecutionContext::for_tests(1);

    c.bench_function("posterior_gcomp_eval_n400_d512", |b| {
        b.iter(|| {
            evaluator
                .evaluate_batch(black_box(&compiled), black_box(batch), &mut out, &mut ws, &ctx)
                .unwrap();
            black_box(out.values[0]);
        });
    });
    assert_eq!(ws.grow_count, grow0, "eval workspace must reuse buffers");
    let _ = CompiledGCompAte;
}

criterion_group!(benches, bench_gcomp_eval);
criterion_main!(benches);
