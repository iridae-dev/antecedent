//! Exact panel source adaptation and execution dependencies.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointPanelError, JointPanelRequest, JointPanelState, JointPanelWorld, engine};
use antecedent_core::execution_attempt::{PhysicalEvent, record_physical};
use antecedent_core::physical_choice::PhysicalChoice;
use antecedent_core::physical_choice::{
    AvailableCapabilities, CostQuotes, DeclaredCost, OperatorCapability, ResourceCapability,
};
use antecedent_core::{CanonicalWriter, Determinism};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_data::PanelData;
use antecedent_data::{ColumnView, SamplingRegularity, TableView};

pub(super) fn check_cancelled(ctx: &ExecutionContext) -> Result<(), JointPanelError> {
    if ctx.cancellation.is_cancelled() { Err(engine::PanelError::Cancelled.into()) } else { Ok(()) }
}
pub(super) fn check_memory(ctx: &ExecutionContext, bytes: u64) -> Result<(), JointPanelError> {
    if ctx.memory_limit_bytes().is_some_and(|limit| bytes > limit) {
        return Err(engine::PanelError::Resource(format!(
            "retained panel allocation needs {bytes} bytes above hard limit"
        ))
        .into());
    }
    Ok(())
}
fn overflow() -> JointPanelError {
    engine::PanelError::Resource("panel size overflow".into()).into()
}
fn add(total: &mut usize, value: usize) -> Result<(), JointPanelError> {
    *total = total.checked_add(value).ok_or_else(overflow)?;
    Ok(())
}
pub(super) fn source_quote(
    source: &PanelData,
    request: &JointPanelRequest,
) -> Result<u64, JointPanelError> {
    let mut bytes = std::mem::size_of::<JointPanelState>();
    add(&mut bytes, std::mem::size_of::<JointPanelRequest>())?;
    for variable in source.schema().variables() {
        add(&mut bytes, std::mem::size_of_val(variable))?;
        add(&mut bytes, variable.name.len())?;
        if let Some(unit) = &variable.unit {
            add(&mut bytes, unit.len())?;
        }
        if let Some(description) = &variable.measurement.description {
            add(&mut bytes, description.len())?;
        }
    }
    for channel in &request.columns.channels {
        add(
            &mut bytes,
            std::mem::size_of::<engine::PanelChannel>() + channel.raw().to_string().len(),
        )?;
    }
    for interaction in &request.interactions {
        let nodes = wire_nodes(&interaction.expression, 1)?;
        let wire_bytes = nodes
            .checked_mul(std::mem::size_of::<antecedent_io::query_wire::ValueExprWire>())
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(overflow)?;
        add(&mut bytes, wire_bytes)?;
        add(&mut bytes, interaction.id.len().checked_mul(2).ok_or_else(overflow)?)?;
        let plan = antecedent_model::program::NodeProgram::column_compile_bytes(
            request.columns.channels.len(),
            nodes,
        )
        .map_err(JointPanelError::compile)?;
        add(&mut bytes, usize::try_from(plan).map_err(|_| overflow())?)?;
    }
    for unit in source.units() {
        add(&mut bytes, unit_quote(unit, request)?)?;
    }
    add(&mut bytes, program_quote(&request.worlds)?)?;
    u64::try_from(bytes).map_err(|_| overflow())
}

fn program_quote(worlds: &[JointPanelWorld]) -> Result<usize, JointPanelError> {
    let mut bytes = 0_usize;
    for world in worlds {
        add(&mut bytes, std::mem::size_of::<JointPanelWorld>())?;
        add(&mut bytes, world.program.id.len())?;
        for channel in &world.program.channels {
            add(&mut bytes, std::mem::size_of::<engine::PanelChannelProgram>())?;
            add(&mut bytes, channel.id.len())?;
            add(&mut bytes, channel.values.len().checked_mul(8).ok_or_else(overflow)?)?;
        }
        for scenario in &world.program.scenarios {
            add(&mut bytes, std::mem::size_of::<engine::PanelScenario>())?;
            add(&mut bytes, scenario.group.len())?;
            for history in &scenario.history {
                add(&mut bytes, std::mem::size_of::<engine::PanelChannelHistory>())?;
                add(&mut bytes, history.id.len())?;
            }
            add(
                &mut bytes,
                scenario
                    .confounder
                    .len()
                    .checked_add(scenario.season.len())
                    .and_then(|n| n.checked_mul(8))
                    .ok_or_else(overflow)?,
            )?;
        }
    }
    Ok(bytes)
}
pub(super) fn retarget_quote(
    request: &JointPanelRequest,
    worlds: &[JointPanelWorld],
    target: &antecedent_core::ScientificTarget,
) -> Result<u64, JointPanelError> {
    let mut bytes = std::mem::size_of::<JointPanelRequest>();
    // One retained program plus lower's simultaneously validated program copies.
    add(&mut bytes, program_quote(worlds)?.checked_mul(2).ok_or_else(overflow)?)?;
    add(
        &mut bytes,
        request
            .columns
            .channels
            .len()
            .checked_mul(std::mem::size_of::<antecedent_core::VariableId>())
            .ok_or_else(overflow)?,
    )?;
    for feature in &request.interactions {
        add(&mut bytes, feature.id.len() + std::mem::size_of_val(feature))?;
        add(
            &mut bytes,
            wire_nodes(&feature.expression, 1)?
                .checked_mul(std::mem::size_of::<antecedent_io::query_wire::ValueExprWire>())
                .ok_or_else(overflow)?,
        )?;
    }
    if let engine::PanelInference::Bayesian { prior, .. } = &request.options.inference {
        add(
            &mut bytes,
            prior.lag.len().checked_mul(std::mem::size_of::<[f64; 3]>()).ok_or_else(overflow)?,
        )?;
    }
    if let Some(initial) = &request.options.initial {
        add(&mut bytes, std::mem::size_of_val(initial))?;
        for channel in &initial.channels {
            add(&mut bytes, std::mem::size_of_val(channel) + channel.id.len())?;
        }
        for feature in &initial.interactions {
            add(&mut bytes, std::mem::size_of_val(feature) + feature.id.len())?;
        }
        add(&mut bytes, initial.season_effects.len().checked_mul(8).ok_or_else(overflow)?)?;
    }
    if let Some(spec) = &request.temporal {
        use engine::time_varying::PanelMeanParameter;
        for parameter in &spec.varying {
            add(&mut bytes, std::mem::size_of_val(parameter))?;
            match parameter {
                PanelMeanParameter::Channel { id } | PanelMeanParameter::Interaction { id } => {
                    add(&mut bytes, id.len())?
                }
                PanelMeanParameter::Lag { channel, .. } => add(&mut bytes, channel.len())?,
                PanelMeanParameter::Intercept
                | PanelMeanParameter::Confounder
                | PanelMeanParameter::Season { .. }
                | PanelMeanParameter::LogHalfSaturation => {}
            }
        }
        add(
            &mut bytes,
            spec.change_points
                .len()
                .checked_add(spec.smooth_knots.len())
                .and_then(|n| n.checked_mul(std::mem::size_of::<usize>()))
                .ok_or_else(overflow)?,
        )?;
        add(&mut bytes, spec.smoothing.len().checked_mul(8).ok_or_else(overflow)?)?;
    }
    add(
        &mut bytes,
        target
            .assumptions()
            .records()
            .len()
            .checked_mul(std::mem::size_of::<antecedent_core::AssumptionRecord>())
            .ok_or_else(overflow)?,
    )?;
    for world in worlds {
        let terms =
            world.horizon().checked_mul(request.columns.channels.len()).ok_or_else(overflow)?;
        let term_width = std::mem::size_of::<antecedent_core::Intervention>()
            + std::mem::size_of::<antecedent_core::SequencedIntervention>()
            + std::mem::size_of::<antecedent_core::Value>();
        add(&mut bytes, terms.checked_mul(term_width).ok_or_else(overflow)?)?;
        add(
            &mut bytes,
            std::mem::size_of::<antecedent_core::InterventionWorld>()
                + std::mem::size_of::<super::JointPanelWorldSupport>(),
        )?;
        for scenario in &world.program.scenarios {
            add(&mut bytes, scenario.group.len() + std::mem::size_of::<String>())?;
        }
    }
    u64::try_from(bytes).map_err(|_| overflow())
}

fn unit_quote(
    unit: &antecedent_data::PanelUnit,
    request: &JointPanelRequest,
) -> Result<usize, JointPanelError> {
    let mut bytes = 0_usize;
    add(&mut bytes, std::mem::size_of::<engine::PanelGroup>())?;
    add(&mut bytes, std::mem::size_of_val(unit))?;
    add(
        &mut bytes,
        request
            .columns
            .channels
            .len()
            .checked_mul(std::mem::size_of::<Vec<f64>>() + std::mem::size_of::<[f64; 2]>())
            .ok_or_else(overflow)?,
    )?;
    add(&mut bytes, unit.unit_id.to_string().len())?;
    let storage = unit.series.storage();
    for column in storage.columns() {
        let view = column.as_view();
        let (n, width) = match view {
            ColumnView::Float64(c) => (c.values.len(), 8),
            ColumnView::Int64(c) => (c.values.len(), 8),
            ColumnView::Boolean(c) => (c.values.len(), 1),
            ColumnView::Timestamp(c) => (c.values_ns.len(), 8),
            ColumnView::Categorical(c) => {
                for level in c.domain.levels.iter() {
                    add(&mut bytes, level.label.len())?;
                }
                (c.codes.len(), 4)
            }
            ColumnView::FixedVector(c) => (c.values.len(), 8),
        };
        add(&mut bytes, n.checked_mul(width).ok_or_else(overflow)?)?;
        add(&mut bytes, view.len().div_ceil(8))?;
    }
    if let Some(weights) = storage.weights() {
        add(&mut bytes, weights.len().checked_mul(8).ok_or_else(overflow)?)?;
    }
    if storage.analysis_mask().is_some() {
        add(&mut bytes, unit.series.row_count().div_ceil(8))?;
    }
    // Actual retained numeric columns and simultaneous coercion buffers.
    let rows = unit.series.row_count();
    add(
        &mut bytes,
        rows.saturating_sub(request.lag_order)
            .checked_mul(request.columns.channels.len().checked_add(3).ok_or_else(overflow)?)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(overflow)?,
    )?;
    for role in request.columns.channels.iter().copied().chain([
        request.columns.outcome,
        request.columns.confounder,
        request.columns.season,
    ]) {
        if !matches!(
            unit.series.column(role).map_err(JointPanelError::compile)?,
            ColumnView::Float64(_)
        ) {
            add(&mut bytes, rows.checked_mul(8).ok_or_else(overflow)?)?;
        }
    }
    Ok(bytes)
}
pub(super) fn adapt(
    source: &PanelData,
    request: &JointPanelRequest,
    ctx: &ExecutionContext,
) -> Result<(engine::PanelData, SemanticDigest), JointPanelError> {
    let c = &request.columns;
    let mut roles = c.channels.clone();
    roles.extend([c.outcome, c.confounder, c.season]);
    if roles.iter().copied().collect::<std::collections::BTreeSet<_>>().len() != roles.len()
        || c.channels.is_empty()
    {
        return Err(JointPanelError::compile("panel roles must be distinct"));
    }
    let mut writer = CanonicalWriter::new("antecedent.compiler.joint_panel.source.v1");
    // Full schema semantics are bound, independently of display-only program labels.
    writer.bytes(
        &antecedent_io::to_cbor(&antecedent_io::schema_to_wire(source.schema()))
            .map_err(JointPanelError::compile)?,
    );
    for role in roles {
        source.schema().get(role).map_err(JointPanelError::compile)?;
        writer.word(u64::from(role.raw()));
    }
    let prefix = request.lag_order;
    if !matches!(prefix, 0 | 2) {
        return Err(JointPanelError::compile("only lag orders zero and two are supported"));
    }
    let mut groups = Vec::with_capacity(source.units().len());
    let mut cadence = None;
    for unit in source.units() {
        groups.push(adapt_group(unit, request, ctx, &mut cadence, &mut writer)?);
    }
    let data = engine::PanelData {
        groups,
        channels: c
            .channels
            .iter()
            .map(|id| engine::PanelChannel { id: id.raw().to_string() })
            .collect(),
        interactions: request.interactions.clone(),
        seasons: request.seasons,
        lag_order: prefix,
    };
    writer.bytes(data.identity()?.as_bytes());
    Ok((data, SemanticDigest::from_bytes(writer.finish())))
}
fn adapt_group(
    unit: &antecedent_data::PanelUnit,
    request: &JointPanelRequest,
    ctx: &ExecutionContext,
    cadence: &mut Option<u64>,
    writer: &mut CanonicalWriter,
) -> Result<engine::PanelGroup, JointPanelError> {
    let c = &request.columns;
    let prefix = request.lag_order;
    check_cancelled(ctx)?;
    let series = &unit.series;
    let interval = match series.time_index().regularity {
        SamplingRegularity::Regular { interval_ns } if interval_ns > 0 => interval_ns,
        _ => {
            return Err(JointPanelError::compile(
                "panel requires explicit positive regular cadence",
            ));
        }
    };
    if cadence.is_some_and(|previous| previous != interval)
        || series.time_index().length != series.row_count()
    {
        return Err(JointPanelError::compile("panel cadence or time-index length differs"));
    }
    *cadence = Some(interval);
    if series.row_count() < prefix + 3 {
        return Err(JointPanelError::compile(
            "panel needs observed prehistory and at least three follow-up rows",
        ));
    }
    if series.storage().weights().is_some() || series.storage().analysis_mask().is_some() {
        return Err(JointPanelError::compile(
            "weighted or masked panel laws require a matching engine and cannot be ignored",
        ));
    }
    writer.word(u64::from(unit.unit_id)).word(interval).bytes(&series.storage().content_digest());
    let channel_values = c
        .channels
        .iter()
        .map(|&id| series.float64_cow(id).map_err(JointPanelError::compile))
        .collect::<Result<Vec<_>, _>>()?;
    let y = series.float64_cow(c.outcome).map_err(JointPanelError::compile)?;
    let z = series.float64_cow(c.confounder).map_err(JointPanelError::compile)?;
    let season = series.float64_cow(c.season).map_err(JointPanelError::compile)?;
    let history = channel_values
        .iter()
        .map(|values| if prefix == 2 { [values[0], values[1]] } else { [0.0; 2] })
        .collect();
    let channels = channel_values.iter().map(|values| values[prefix..].to_vec()).collect();
    let mut levels = Vec::with_capacity(series.row_count() - prefix);
    for &value in &season[prefix..] {
        if !value.is_finite()
            || value < 0.0
            || value.fract() != 0.0
            || value >= request.seasons as f64
        {
            return Err(JointPanelError::compile("season must be a finite declared integer level"));
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "finite nonnegative integer strictly below declared usize season count checked above"
        )]
        let level = value as usize;
        levels.push(level);
    }
    let group = engine::PanelGroup {
        id: unit.unit_id.to_string(),
        history,
        channels,
        outcome: y[prefix..].to_vec(),
        confounder: z[prefix..].to_vec(),
        season: levels,
    };
    record_physical(PhysicalEvent::Scan, (c.channels.len() + 3) as u64);
    record_physical(PhysicalEvent::Copy, (c.channels.len() + 3) as u64);
    record_physical(
        PhysicalEvent::CopiedBytes,
        ((series.row_count() - prefix) * (c.channels.len() + 3) * 8) as u64,
    );
    Ok(group)
}
pub(super) fn physical_choice(data: &engine::PanelData) -> Result<PhysicalChoice, JointPanelError> {
    let rows = data.groups.iter().map(|group| group.outcome.len() as u64).sum::<u64>();
    let cost = DeclaredCost::from_raw(rows.saturating_mul(
        (data.seasons + 8 + data.channels.len() * 4 + data.interactions.len()) as u64,
    ));
    PhysicalChoice::choose(
        CostQuotes::new(cost, cost, cost, cost),
        AvailableCapabilities::empty()
            .with_operator(OperatorCapability::PreparedNumericView)
            .with_resource(ResourceCapability::PreparedColumns),
    )
    .map_err(JointPanelError::compile)
}
pub(super) fn fit_dependency(
    state: &JointPanelState,
    ctx: &ExecutionContext,
) -> Result<SemanticDigest, JointPanelError> {
    let mut writer = CanonicalWriter::new("antecedent.compiler.joint_panel.fit.v1");
    writer
        .bytes(state.source_identity.as_bytes())
        .bytes(state.options_identity.as_bytes())
        .word(ctx.rng.master_seed())
        .word(u64::from(matches!(ctx.determinism, Determinism::Strict)))
        .word(u64::from(ctx.kernel_policy.allow_portable_optimized))
        .word(u64::from(ctx.kernel_policy.allow_arch_simd))
        .word(u64::from(ctx.kernel_policy.force_scalar));
    if let Some(spec) = &state.request.temporal {
        writer.word(1).bytes(spec.identity(&state.data)?.as_bytes());
    } else {
        writer.word(0);
    }
    Ok(SemanticDigest::from_bytes(writer.finish()))
}
pub(super) fn observe_reuse() {
    record_physical(PhysicalEvent::CacheHit, 1);
}

fn wire_nodes(
    expression: &antecedent_io::query_wire::ValueExprWire,
    depth: usize,
) -> Result<usize, JointPanelError> {
    use antecedent_io::query_wire::ValueExprWire as Expr;
    if depth > antecedent_core::value_expr::MAX_EXPR_DEPTH {
        return Err(JointPanelError::compile(
            "interaction expression exceeds the common depth bound",
        ));
    }
    let next = depth + 1;
    let children = match expression {
        Expr::Const { .. } | Expr::Generated | Expr::Var { .. } | Expr::Lag { .. } => 0,
        Expr::Neg { operand } | Expr::Not { operand } | Expr::Indicator { operand } => {
            wire_nodes(operand, next)?
        }
        Expr::Arith { lhs, rhs, .. }
        | Expr::Compare { lhs, rhs, .. }
        | Expr::Logic { lhs, rhs, .. } => {
            wire_nodes(lhs, next)?.checked_add(wire_nodes(rhs, next)?).ok_or_else(overflow)?
        }
        Expr::Clip { value, lower, upper } => {
            let a = wire_nodes(value, next)?;
            let b = wire_nodes(lower, next)?;
            let c = wire_nodes(upper, next)?;
            a.checked_add(b).and_then(|n| n.checked_add(c)).ok_or_else(overflow)?
        }
        Expr::If { condition, then_branch, else_branch } => {
            let a = wire_nodes(condition, next)?;
            let b = wire_nodes(then_branch, next)?;
            let c = wire_nodes(else_branch, next)?;
            a.checked_add(b).and_then(|n| n.checked_add(c)).ok_or_else(overflow)?
        }
        Expr::Callback { .. } => {
            return Err(JointPanelError::compile(
                "panel features require callback-free common expressions",
            ));
        }
    };
    let nodes = children.checked_add(1).ok_or_else(overflow)?;
    if nodes > antecedent_core::value_expr::MAX_EXPR_NODES {
        return Err(JointPanelError::compile(
            "interaction expression exceeds the common node bound",
        ));
    }
    Ok(nodes)
}
