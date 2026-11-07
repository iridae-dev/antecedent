use super::{
    Arc, Assignment, BTreeMap, BoundTransportFunctional, EmpiricalTableOptions, EstimationError,
    ExactEvaluationLimits, ExecutionContext, ReplicateColumns, ResamplingPlan,
    StatisticalTransportInput, StreamDomain, TransportStream, assemble_statistical_laws,
    bound_sample_key, fill_resample_indexes, is_support_failure, prepare_exact_transport,
    refuse_eval,
};

#[derive(Clone)]
pub(super) struct BootstrapDraws {
    pub(super) ids: Arc<[u32]>,
    pub(super) ok: u32,
    pub(super) failed: u32,
    pub(super) columns: ReplicateColumns,
}

pub(super) fn outer_bootstrap(
    functional: &BoundTransportFunctional,
    input: &StatisticalTransportInput,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    options: &EmpiricalTableOptions,
    ctx: &ExecutionContext,
) -> Result<Vec<BootstrapDraws>, EstimationError> {
    let outcomes = functional.derivation().query().outcomes.clone();
    let mut columns: Vec<ReplicateColumns> =
        requests.iter().map(|_| ReplicateColumns::new(&outcomes)).collect();
    let mut ids = Vec::new();
    let mut failed = 0u32;
    // Datasets are resampled in the order of their bound sample keys, whatever
    // the order the evidence was supplied in, so the same evidence always draws
    // the same replicates; forwarded aliases share one dataset.
    let mut samples: Vec<_> = input
        .samples
        .iter()
        .map(|sample| (bound_sample_key(functional.catalog(), sample), sample))
        .collect();
    samples.sort_by(|a, b| a.0.cmp(&b.0));
    samples.dedup_by(|a, b| a.0 == b.0);
    // Every replicate resamples each dataset on its own `(dataset, replicate)` stream, so
    // replicates are independent; they are evaluated across the context's thread budget
    // and folded in index order. `None` is a counted failed replicate.
    let attempts = ctx.map_indexed::<_, EstimationError, _>(
        options.bootstrap_replicates as usize,
        |index, ctx| {
            let replicate = u32::try_from(index).unwrap_or(u32::MAX);
            crate::transport::refuse_cancelled(ctx, "transport statistical execution")?;
            let mut indexes = Vec::new();
            let mut row_indexes = BTreeMap::new();
            for (dataset, (key, sample)) in samples.iter().enumerate() {
                let n = sample.n();
                if n == 0 {
                    return Ok(None);
                }
                let mut rng = ctx.rng.stream_for(
                    StreamDomain::Transport,
                    TransportStream::Bootstrap.index(dataset, replicate)?,
                );
                indexes.clear();
                if antecedent_core::DeclaredSamplingDesign::IndependentSubjectTrajectories
                    .apply_resampling(antecedent_core::ResamplingRule::RowSubjectIid)
                    .is_err()
                    || fill_resample_indexes(
                        ResamplingPlan::IidBootstrap,
                        n,
                        &mut rng,
                        &mut indexes,
                    )
                    .is_err()
                {
                    return Ok(None);
                }
                row_indexes.insert(key.clone(), indexes.clone());
            }
            let assembled =
                match assemble_statistical_laws(input, functional, options, &row_indexes, ctx) {
                    Ok(data) => {
                        data.with_shared_factor_cache(if requests.len() > 1 { 1024 } else { 0 })
                    }
                    Err(EstimationError::EmptyEmpiricalSample { .. }) => return Ok(None),
                    Err(error) => return Err(error),
                };
            let mut complete = Vec::with_capacity(requests.len());
            for request in requests {
                match prepare_exact_transport(
                    functional,
                    assembled.clone(),
                    request.clone(),
                    limits,
                    ctx,
                )
                .and_then(|plan| plan.evaluate(ctx))
                {
                    Ok(distribution) => complete.push(distribution),
                    Err(error) => {
                        crate::transport::refuse_cancelled(ctx, "transport statistical execution")?;
                        if is_support_failure(&error) {
                            break;
                        }
                        return Err(refuse_eval(&error));
                    }
                }
            }
            if let Some(progress) = &ctx.progress {
                progress.report(
                    f64::from(replicate + 1) / f64::from(options.bootstrap_replicates),
                    "transport.bootstrap",
                );
            }
            Ok((complete.len() == requests.len()).then_some(complete))
        },
    )?;
    for (index, attempt) in attempts.into_iter().enumerate() {
        let Some(complete) = attempt else {
            failed += 1;
            continue;
        };
        for (column, distribution) in columns.iter_mut().zip(&complete) {
            column.record(distribution)?;
        }
        ids.push(u32::try_from(index).unwrap_or(u32::MAX));
    }
    crate::transport::refuse_cancelled(ctx, "transport statistical execution")?;
    let ids: Arc<[u32]> = ids.into();
    let ok = u32::try_from(ids.len()).unwrap_or(u32::MAX);
    Ok(columns
        .into_iter()
        .map(|columns| BootstrapDraws { ids: Arc::clone(&ids), ok, failed, columns })
        .collect())
}
