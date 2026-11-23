//! Statistical artifact uncertainty and posterior validation.

use super::{
    Arc, BayesianTransportPosteriorWire, ExactLawWire, ExactTransportData, IntervalWithholdReason,
    IoError, PERCENTILE_FLOOR, PosteriorIntervalMethodWire, PreparedStudy, ReplicatePolicy,
    SampleSummary, StatisticalExecutionWire, StatisticalPreparedState,
    StatisticalTransportEstimate, UncertaintyRowWire, VariableId, err, is_bayesian_provider,
    is_probability_vector,
};

/// Why a statistical artifact's uncertainty section fails to match its prepared study.
///
/// Each variant names one concern; [`Self::into_error`] maps it to the refusal published to
/// callers, so the checks stay small and the observable messages stay the same.
#[derive(Debug)]
pub(super) enum UncertaintyFault {
    /// The absent-row artifact claims intervals, replicates or a reason the study cannot earn.
    AbsentRowClaims,
    /// The row's estimator, coverage, seed, sample sizes or replicate counts disagree.
    RowMetadata,
    /// The evidence catalog does not support iid resampling of the sampled regimes.
    UnsupportedDependence,
    /// The replicate matrix or its ids are missing or malformed.
    ReplicateShape,
    /// A replicate row is not a probability law over the fitted atoms.
    ReplicateLaw,
    /// The replicates exceed the artifact's operation budget.
    OperationBudget,
    /// The withheld or published intervals and reason disagree with the replicates.
    IntervalClaims,
    /// A lower-level conversion failure, published unchanged.
    Propagated(IoError),
}

impl UncertaintyFault {
    fn into_error(self) -> IoError {
        match self {
            Self::AbsentRowClaims
            | Self::RowMetadata
            | Self::UnsupportedDependence
            | Self::ReplicateShape
            | Self::ReplicateLaw
            | Self::IntervalClaims => err("transport artifact uncertainty mismatch"),
            Self::OperationBudget => err("transport artifact operation budget"),
            Self::Propagated(error) => error,
        }
    }
}

/// `Ok` when `holds`, otherwise the named fault.
fn ensure<F>(holds: bool, fault: F) -> Result<(), F> {
    if holds { Ok(()) } else { Err(fault) }
}

pub(super) fn validate_uncertainty(
    wire: &StatisticalExecutionWire,
    point: antecedent_expr::ExactDistribution,
    prepared: &PreparedStudy<StatisticalPreparedState>,
) -> Result<StatisticalTransportEstimate, IoError> {
    check_uncertainty(wire, point, prepared).map_err(UncertaintyFault::into_error)
}

fn check_uncertainty(
    wire: &StatisticalExecutionWire,
    point: antecedent_expr::ExactDistribution,
    prepared: &PreparedStudy<StatisticalPreparedState>,
) -> Result<StatisticalTransportEstimate, UncertaintyFault> {
    let mut estimate = StatisticalTransportEstimate {
        distribution: point,
        uncertainty: None,
        atom_intervals: None,
        mean_intervals: None,
        atom_replicates: None,
        mean_replicates: None,
        replicate_ids: None,
        uncertainty_reason: wire.uncertainty_reason.map(|reason| Arc::from(reason.as_str())),
    };
    let Some(row) = &wire.uncertainty else {
        check_absent_uncertainty(wire, prepared)?;
        return Ok(estimate);
    };
    check_uncertainty_row(wire, row)?;
    ensure(
        antecedent_estimate::supported_iid_regimes(
            prepared.evidence_catalog(),
            wire.samples.iter().map(|sample| antecedent_core::RegimeId::from_raw(sample.regime)),
        ),
        UncertaintyFault::UnsupportedDependence,
    )?;
    let rows = wire.atom_replicates.as_ref().ok_or(UncertaintyFault::ReplicateShape)?;
    let ids = wire.replicate_ids.as_ref().ok_or(UncertaintyFault::ReplicateShape)?;
    check_replicate_ids(row, rows.len(), ids)?;
    let width = estimate.distribution.atoms.len();
    ensure(
        rows.len().checked_mul(width).is_some_and(|n| n <= wire.operations),
        UncertaintyFault::OperationBudget,
    )?;
    let means = replicate_means(&estimate.distribution, rows)?;
    if let Some((atoms, intervals)) = check_interval_claims(wire, row, rows, &means, width)? {
        estimate.atom_intervals = Some(atoms.into());
        estimate.mean_intervals = Some(
            intervals.into_iter().map(|(v, lo, hi)| (VariableId::from_raw(v), lo, hi)).collect(),
        );
    }
    estimate.uncertainty = Some(row.to_row());
    estimate.atom_replicates = Some(rows.iter().map(|row| Arc::from(row.clone())).collect());
    estimate.mean_replicates = Some(means.into_iter().map(|(v, row)| (v, row.into())).collect());
    estimate.replicate_ids = Some(ids.clone().into());
    Ok(estimate)
}

/// The reason a study without an uncertainty row must carry.
fn absent_uncertainty_reason(
    wire: &StatisticalExecutionWire,
) -> Result<IntervalWithholdReason, IoError> {
    Ok(if wire.samples.is_empty() {
        IntervalWithholdReason::ExactSuppliedLaw
    } else if is_bayesian_provider(wire.options.to_options()?.estimator)
        || wire.options.bootstrap_replicates == 0
    {
        IntervalWithholdReason::BootstrapNotRequested
    } else {
        IntervalWithholdReason::UnsupportedDependence
    })
}

/// No uncertainty row: no intervals or replicates may ride along, the study must carry the
/// reason its kind implies, and the facade must not have supported an interval.
fn check_absent_uncertainty(
    wire: &StatisticalExecutionWire,
    prepared: &PreparedStudy<StatisticalPreparedState>,
) -> Result<(), UncertaintyFault> {
    let expected = absent_uncertainty_reason(wire)?;
    ensure(
        wire.atom_intervals.is_none()
            && wire.mean_intervals.is_none()
            && wire.atom_replicates.is_none()
            && wire.replicate_ids.is_none()
            && wire.uncertainty_reason == Some(expected)
            && !prepared.supported_interval(),
        UncertaintyFault::AbsentRowClaims,
    )
}

/// The row's provenance and replicate bookkeeping against the study's declared options.
///
/// The interval method, scope and support regime are single-variant wire types, so a row
/// naming any other value never reaches this check.
fn check_uncertainty_row(
    wire: &StatisticalExecutionWire,
    row: &UncertaintyRowWire,
) -> Result<(), UncertaintyFault> {
    let sizes: Vec<_> = wire.samples.iter().map(|s| (s.snapshot.clone(), s.n)).collect();
    ensure(
        row.estimator == wire.options.estimator
            && row.coverage_target.to_bits() == wire.options.coverage_level.to_bits()
            && row.seed == wire.seed
            && row.sample_sizes == sizes
            && row.calibration_binding.is_none()
            && !wire.samples.is_empty(),
        UncertaintyFault::RowMetadata,
    )?;
    ensure(
        row.replicates_requested == wire.options.bootstrap_replicates
            && row.replicates_requested != 0
            && row.replicates_ok.checked_add(row.replicates_failed)
                == Some(row.replicates_requested),
        UncertaintyFault::RowMetadata,
    )
}

/// Replicate ids are the strictly increasing requested indices of the successful replicates.
fn check_replicate_ids(
    row: &UncertaintyRowWire,
    replicate_rows: usize,
    ids: &[u32],
) -> Result<(), UncertaintyFault> {
    ensure(
        replicate_rows == row.replicates_ok as usize
            && ids.len() == replicate_rows
            && ids.windows(2).all(|pair| pair[0] < pair[1])
            && ids.iter().all(|id| *id < row.replicates_requested),
        UncertaintyFault::ReplicateShape,
    )
}

/// Each replicate row as a law over the fitted atoms, and its outcome means.
fn replicate_means(
    distribution: &antecedent_expr::ExactDistribution,
    rows: &[Vec<f64>],
) -> Result<Vec<(VariableId, Vec<f64>)>, UncertaintyFault> {
    let width = distribution.atoms.len();
    let mut means: Vec<(VariableId, Vec<f64>)> =
        distribution.outcomes.iter().map(|v| (*v, Vec::new())).collect();
    for row in rows {
        ensure(row.len() == width && is_probability_vector(row), UncertaintyFault::ReplicateLaw)?;
        let mut replicate = distribution.clone();
        replicate.probabilities = row.clone().into();
        for (v, values) in &mut means {
            values.push(replicate.mean(*v).map_err(err)?);
        }
    }
    Ok(means)
}

/// The recomputed per-atom intervals and per-outcome mean intervals (raw variable id).
type PublishedIntervals = (Vec<(f64, f64)>, Vec<(u32, f64, f64)>);

/// Check the intervals and withholding reason against the replicates.
///
/// Two distinct reasons withhold the percentile edges: the estimator's own failure-fraction
/// refusal (too few successful replicates or too many failed ones to trust the resample),
/// and the facade's support floor (enough replicates ran clean, but fewer than the
/// nominal-0.95 percentile requires). `withhold_unearned_percentile` never overwrites an
/// existing failure-fraction reason, so the two are mutually exclusive and must be told apart
/// here too. Returns the recomputed intervals when they are published.
fn check_interval_claims(
    wire: &StatisticalExecutionWire,
    row: &UncertaintyRowWire,
    rows: &[Vec<f64>],
    means: &[(VariableId, Vec<f64>)],
    width: usize,
) -> Result<Option<PublishedIntervals>, UncertaintyFault> {
    use antecedent_estimate::percentile_interval;
    let no_intervals = wire.atom_intervals.is_none() && wire.mean_intervals.is_none();
    let counts = (row.replicates_requested, row.replicates_ok, row.replicates_failed);
    if let Err(reason) = ReplicatePolicy::BOOTSTRAP.decide(counts.0, counts.1, counts.2) {
        let expected = IntervalWithholdReason::parse(reason);
        return ensure(
            expected.is_some() && wire.uncertainty_reason == expected && no_intervals,
            UncertaintyFault::IntervalClaims,
        )
        .map(|()| None);
    }
    if PERCENTILE_FLOOR.decide(counts.0, counts.1, counts.2).is_err() {
        let floor = Some(IntervalWithholdReason::InsufficientBootstrapReplicates);
        return ensure(
            wire.uncertainty_reason == floor && no_intervals,
            UncertaintyFault::IntervalClaims,
        )
        .map(|()| None);
    }
    let atoms = (0..width)
        .map(|i| {
            percentile_interval(&rows.iter().map(|r| r[i]).collect::<Vec<_>>(), row.coverage_target)
        })
        .collect::<Option<Vec<_>>>();
    let intervals = means
        .iter()
        .map(|(v, reps)| {
            percentile_interval(reps, row.coverage_target).map(|(lo, hi)| (v.raw(), lo, hi))
        })
        .collect::<Option<Vec<_>>>();
    match (atoms, intervals) {
        (Some(atoms), Some(intervals)) => {
            ensure(
                wire.uncertainty_reason.is_none()
                    && wire.atom_intervals.as_ref() == Some(&atoms)
                    && wire.mean_intervals.as_ref() == Some(&intervals),
                UncertaintyFault::IntervalClaims,
            )?;
            Ok(Some((atoms, intervals)))
        }
        _ => ensure(
            wire.uncertainty_reason == Some(IntervalWithholdReason::NumericalFailure)
                && no_intervals,
            UncertaintyFault::IntervalClaims,
        )
        .map(|()| None),
    }
}

/// Most draw-by-atom probabilities a Bayesian posterior artifact may carry.
const MAX_POSTERIOR_DRAW_CELLS: usize = 10_000_000;

/// Why a Bayesian posterior section fails to match its prepared study.
///
/// Each variant names one concern; [`Self::into_error`] maps it to the refusal published to
/// callers.
#[derive(Debug)]
pub(super) enum PosteriorFault {
    /// The study uses a Bayesian provider but the artifact carries no draws.
    MissingDraws,
    /// Estimator, method, draw counts or size disagree with the study's declaration.
    DrawMetadata,
    /// The reason withholds an interval the draw counts have earned.
    WithholdsEarnedInterval,
    /// A draw is not a probability law over the fitted atoms.
    InvalidDrawLaw,
    /// The published intervals disagree with the draws.
    SummaryMismatch,
    /// A lower-level conversion failure, published unchanged.
    Propagated(IoError),
}

impl PosteriorFault {
    fn into_error(self) -> IoError {
        match self {
            Self::MissingDraws => err("Bayesian transport artifact lacks posterior draws"),
            Self::DrawMetadata => err("Bayesian transport artifact draw metadata mismatch"),
            Self::WithholdsEarnedInterval => {
                err("Bayesian transport artifact withholds an earned posterior interval")
            }
            Self::InvalidDrawLaw => err("Bayesian transport artifact has invalid draw law"),
            Self::SummaryMismatch => err("Bayesian transport artifact posterior summary mismatch"),
            Self::Propagated(error) => error,
        }
    }
}

pub(super) fn validate_bayesian_posterior(
    wire: Option<&BayesianTransportPosteriorWire>,
    point: &antecedent_expr::ExactDistribution,
    estimator: &str,
    requested: u32,
    coverage: f64,
) -> Result<Option<antecedent_estimate::BayesianStatisticalTransportEstimate>, IoError> {
    check_bayesian_posterior(wire, point, estimator, requested, coverage)
        .map_err(PosteriorFault::into_error)
}

fn check_bayesian_posterior(
    wire: Option<&BayesianTransportPosteriorWire>,
    point: &antecedent_expr::ExactDistribution,
    estimator: &str,
    requested: u32,
    coverage: f64,
) -> Result<Option<antecedent_estimate::BayesianStatisticalTransportEstimate>, PosteriorFault> {
    let expected = matches!(
        estimator,
        antecedent_estimate::EMPIRICAL_SUPPORT_BAYESIAN_BOOTSTRAP
            | antecedent_estimate::STATE_SPACE_DIRICHLET
    );
    let Some(wire) = wire else {
        return if expected { Err(PosteriorFault::MissingDraws) } else { Ok(None) };
    };
    check_posterior_metadata(wire, point, estimator, requested, expected)?;
    let reason = wire.interval_reason;
    let parsed = reason.and_then(|reason| IntervalWithholdReason::parse(reason.as_str()));
    let dependence = parsed.is_some_and(IntervalWithholdReason::is_dependence_refusal);
    let withheld = parsed.is_some_and(IntervalWithholdReason::withholds_posterior_summary);
    check_posterior_draw_counts(wire, parsed, dependence)?;
    ensure(
        !withheld || (wire.atom_intervals.is_empty() && wire.mean_intervals.is_empty()),
        PosteriorFault::SummaryMismatch,
    )?;
    let distributions = draw_distributions(wire, point)?;
    if withheld {
        return Ok(Some(antecedent_estimate::BayesianStatisticalTransportEstimate {
            estimator: Arc::from(wire.estimator.as_str()),
            interval_method: Arc::from(wire.interval_method.as_str()),
            distributions: distributions.into(),
            atom_intervals: Arc::from([]),
            mean_intervals: Arc::from([]),
            interval_reason: reason,
            draws_requested: wire.draws_requested,
            draws_ok: wire.draws_ok,
            draws_failed: wire.draws_failed,
        }));
    }
    // A published posterior interval must have earned the facade's draw floor;
    // the producer withholds it below the floor, and so does the consumer.
    ensure(
        reason == Some(antecedent_estimate::TransportIntervalReason::NotMeasured)
            && PERCENTILE_FLOOR
                .decide(wire.draws_requested, wire.draws_ok, wire.draws_failed)
                .is_ok(),
        PosteriorFault::DrawMetadata,
    )?;
    let (atom_intervals, mean_intervals) = posterior_intervals(&distributions, point, coverage)?;
    ensure(
        same_intervals(&atom_intervals, &wire.atom_intervals)
            && mean_intervals.len() == wire.mean_intervals.len()
            && mean_intervals.iter().zip(&wire.mean_intervals).all(
                |((v, lo, hi), (wv, wlo, whi))| {
                    v.raw() == *wv && lo.to_bits() == wlo.to_bits() && hi.to_bits() == whi.to_bits()
                },
            ),
        PosteriorFault::SummaryMismatch,
    )?;
    Ok(Some(antecedent_estimate::BayesianStatisticalTransportEstimate {
        estimator: Arc::from(wire.estimator.as_str()),
        interval_method: Arc::from(wire.interval_method.as_str()),
        distributions: distributions.into(),
        atom_intervals: atom_intervals.into(),
        mean_intervals: mean_intervals.into(),
        draws_requested: wire.draws_requested,
        draws_ok: wire.draws_ok,
        draws_failed: wire.draws_failed,
        interval_reason: Some(antecedent_estimate::TransportIntervalReason::NotMeasured),
    }))
}

/// The artifact's declared estimator, interval method, draw request and size.
fn check_posterior_metadata(
    wire: &BayesianTransportPosteriorWire,
    point: &antecedent_expr::ExactDistribution,
    estimator: &str,
    requested: u32,
    estimator_is_bayesian: bool,
) -> Result<(), PosteriorFault> {
    ensure(
        estimator_is_bayesian
            && wire.estimator == estimator
            && wire.interval_method == PosteriorIntervalMethodWire::EqualTail
            && wire.draws_requested == requested
            && wire.probabilities.len() == wire.draws_ok as usize
            && wire
                .probabilities
                .len()
                .checked_mul(point.atoms.len())
                .is_some_and(|cells| cells <= MAX_POSTERIOR_DRAW_CELLS),
        PosteriorFault::DrawMetadata,
    )
}

/// Draw counts: a dependence refusal carries no draws at all, otherwise the successful and
/// failed draws partition the request. An insufficient-draws reason must not name a request
/// that has in fact earned the floor.
fn check_posterior_draw_counts(
    wire: &BayesianTransportPosteriorWire,
    reason: Option<IntervalWithholdReason>,
    dependence: bool,
) -> Result<(), PosteriorFault> {
    if reason == Some(IntervalWithholdReason::InsufficientBootstrapReplicates)
        && PERCENTILE_FLOOR.decide(wire.draws_requested, wire.draws_ok, wire.draws_failed).is_ok()
    {
        return Err(PosteriorFault::WithholdsEarnedInterval);
    }
    let counts_hold = if dependence {
        wire.draws_ok == 0 && wire.draws_failed == 0 && wire.probabilities.is_empty()
    } else {
        wire.draws_ok.checked_add(wire.draws_failed) == Some(wire.draws_requested)
    };
    ensure(counts_hold, PosteriorFault::DrawMetadata)
}

/// Each carried draw as an exact law over the point distribution's atoms.
fn draw_distributions(
    wire: &BayesianTransportPosteriorWire,
    point: &antecedent_expr::ExactDistribution,
) -> Result<Vec<antecedent_expr::ExactDistribution>, PosteriorFault> {
    let mut distributions = Vec::with_capacity(wire.probabilities.len());
    for probabilities in &wire.probabilities {
        ensure(
            probabilities.len() == point.atoms.len() && is_probability_vector(probabilities),
            PosteriorFault::InvalidDrawLaw,
        )?;
        distributions.push(antecedent_expr::ExactDistribution {
            outcomes: point.outcomes.clone(),
            atoms: point.atoms.clone(),
            probabilities: probabilities.clone().into(),
            support: point.support.clone(),
        });
    }
    Ok(distributions)
}

/// Equal-tail posterior intervals for every atom probability and outcome mean.
#[allow(clippy::type_complexity, reason = "the signature describes the existing typed operation")]
fn posterior_intervals(
    distributions: &[antecedent_expr::ExactDistribution],
    point: &antecedent_expr::ExactDistribution,
    coverage: f64,
) -> Result<(Vec<(f64, f64)>, Vec<(VariableId, f64, f64)>), PosteriorFault> {
    let mut atom_intervals = Vec::with_capacity(point.atoms.len());
    for atom in 0..point.atoms.len() {
        let mut values =
            distributions.iter().map(|draw| draw.probabilities[atom]).collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        atom_intervals.push(antecedent_stats::equal_tail_interval_sorted(
            &values,
            coverage,
            antecedent_stats::QuantileRule::Interpolated,
        ));
    }
    let mut mean_intervals = Vec::with_capacity(point.outcomes.len());
    for outcome in point.outcomes.iter().copied() {
        let mut values = distributions
            .iter()
            .map(|draw| draw.mean(outcome).map_err(err))
            .collect::<Result<Vec<_>, _>>()?;
        values.sort_by(f64::total_cmp);
        let (lower, upper) = antecedent_stats::equal_tail_interval_sorted(
            &values,
            coverage,
            antecedent_stats::QuantileRule::Interpolated,
        );
        mean_intervals.push((outcome, lower, upper));
    }
    Ok((atom_intervals, mean_intervals))
}

fn same_intervals(left: &[(f64, f64)], right: &[(f64, f64)]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits())
}

pub(in crate::analysis) fn validate_sample_summaries(
    samples: &[SampleSummary],
    data: &ExactTransportData,
    catalog: &antecedent_core::EvidenceCatalog,
) -> Result<(), IoError> {
    let mut seen = std::collections::BTreeSet::new();
    for sample in samples {
        antecedent_io::external_estimate::parse_digest_hex(&sample.content_digest)?;
        let key =
            antecedent_io::to_cbor(&(&sample.population, sample.regime, &sample.interventions))?;
        if sample.n == 0 || !seen.insert(key) {
            return Err(err("invalid empirical sample summary"));
        }
        let law = data
            .laws()
            .iter()
            .find(|law| {
                let wire = ExactLawWire::metadata(law);
                wire.population == sample.population
                    && wire.regime == sample.regime
                    && wire.interventions == sample.interventions
            })
            .ok_or_else(|| err("sample summary has no fitted joint"))?;
        if law.origin() == antecedent_expr::LawOrigin::SuppliedExact
            || law.snapshot_identity() != sample.snapshot
        {
            return Err(err("sample summary origin/snapshot mismatch"));
        }
        if catalog.bindings.iter().any(|b| {
            b.regime.raw() == sample.regime && b.snapshot_identity.as_ref() != sample.snapshot
        }) {
            return Err(err("sample summary binding mismatch"));
        }
        if let Some(counts) = law.empirical_counts() {
            if counts.iter().sum::<u64>() != u64::from(sample.n) {
                return Err(err("learned support/sample size mismatch"));
            }
            continue;
        }
        for p in law.probabilities() {
            let count = p * f64::from(sample.n);
            if (count - count.round()).abs() > (8.0 * f64::EPSILON * f64::from(sample.n)).max(1e-7)
            {
                return Err(err("fitted joint is inconsistent with sample size"));
            }
        }
    }
    let mut aliases = std::collections::BTreeMap::new();
    for sample in samples {
        if let Some(identity) = catalog
            .bindings
            .iter()
            .find(|binding| binding.regime.raw() == sample.regime)
            .and_then(|binding| binding.dataset_identity.as_ref())
        {
            let key = antecedent_io::to_cbor(&(identity.as_ref(), &sample.interventions))?;
            if let Some(previous) = aliases.insert(key, sample)
                && (previous.n != sample.n || previous.content_digest != sample.content_digest)
            {
                return Err(err("conflicting forwarded dataset provenance"));
            }
        }
    }
    let empirical = data
        .laws()
        .iter()
        .filter(|law| law.origin() != antecedent_expr::LawOrigin::SuppliedExact)
        .count();
    if empirical != samples.len() {
        return Err(err("fitted joints lack sample provenance"));
    }
    Ok(())
}
