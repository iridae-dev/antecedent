# 2.3 A0 identity and coupling contract

This contract applies to new 2.3 scientific claims and artifacts. A digest of one field never substitutes for another. Each promoted route must name the subset it consumes and carry those values through its public producer and independent consumer. The 2.2 wire formats keep their existing identities.

| Identity | Required meaning | Invalidation |
| --- | --- | --- |
| `source_population` / `target_population` | Stable, distinct population IDs and the direction of transport. A same-population query still names both roles explicitly. | A source/target swap or new population invalidates transport and prior mappings. |
| `graph` | Canonical graph kind, nodes and edges, with completion/atom ID when applicable. | Edge, node, latent or completion changes require a new identification decision. |
| `selection` | Named source-to-target selection targets or sampling-selection variables, including their semantics. | Changing a target or interpreting population difference as selection bias invalidates the proof. |
| `regime` | Intervention/policy/conditioning assignments and their time order, separately for every source and target factor. | A changed joint regime invalidates the cited factor; separate regimes cannot be merged into one joint law. |
| `time` | Horizon, ordered period boundaries, pre-action history and outcome time. | A new period, shifted window or third slice invalidates a two-slice proof until replayed. |
| `observation` | Snapshot, dataset, unit/row ownership and missingness pattern; factor identities identify the actual measured law. | A refresh or recovered-pattern change invalidates fitted factors and dependent summaries. |
| `sampling` | Independent-study, shared-unit, cluster or repeated-unit design, including resampling unit and declared cross-source dependence. | An altered unit map or unknown dependence withholds its sampling interval. |
| `provider` | Native implementation ID or external provider fingerprint, capability and trust state. | A new provider/request fingerprint loses exact-request verification and requires rebinding. |

The canonical claim identity binds the scientific fields above and its estimand. The data identity binds observation and sampling fields. A numerical execution identity additionally binds provider, algorithm options, seed, limits and uncertainty method. A refreshed snapshot may reuse an identified formula only after the formula's graph, selection, regime, population and time premises are checked again; it cannot reuse an old estimate or interval.

## Coupling rules

- **Shared data across graph scenarios:** each scenario retains its own graph and selection identity. Reused observations retain the same dataset, snapshot and unit IDs. Covariance or a whole-method replicate uses one unit selection per replicate across every evaluated scenario; independent scenario resamples are a different design.
- **Repeated units through time:** all periods from one unit stay in one fitting fold and one resampling unit. A time extension preserves unit lineage but changes time and observation identities. Its old posterior/interval cannot be relabeled for the new period.
- **Aligned posterior or sampling draws:** draw `i` is one joint realization only when every coordinate shares the same declared draw-set ID, draw meaning, seed/chain or replicate ID, provider and source snapshot. Independently generated marginal arrays never become a joint law by equal length or row order. Posterior parameter, posterior predictive, bootstrap and structural scenario draws have distinct meanings.
- **Cross-world histories:** both potential-outcome worlds for one unit read one abduced exogenous history. The unit and exogenous-draw IDs stay identical across worlds; interventions and world IDs differ. A transported counterfactual additionally binds the target-population and selection proof rather than relabeling a source-world draw.
- **Evidence and inference:** a structural envelope stores completion status and mass separately from any posterior weight. Unidentified and unevaluated mass remain visible; normalization over successful atoms would change the claim. A sampling interval binds its own design and calibration coordinate. An external value retains attested or exact-request-verified trust; neither state becomes native licensing.

Every 2.3 artifact consumer must compare the expected identity fields before using values, enforce size and work bounds before decoding or replay, and refuse a resealed semantic mutation. The initial frozen promotion cells in `promotion_2_3.toml` name their relevant identity inputs; this contract adds no public license by itself.
