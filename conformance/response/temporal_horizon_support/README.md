# Horizon-varying empirical support

Same lag-1 / lag-2 outcome template as `temporal_dose_horizon`, different
treatment process. For `n = 80`,

```text
T_s = 10  if s ≥ n-7 else  0.05 sin(s)
Y_s = 1 + 2 T_{s-1} + 3 T_{s-2}
```

The policy is `Pulse { at: -1 }` at horizons `[1, 8]`. A longer-horizon pulse
looks further back, so the late-series spike of `10` is inside the horizon-1
lag-aligned treatment column and absent from horizon 8.

Requested doses `[0, 5]`:

| cell | status |
|---|---|
| dose 0, h=1 | supported |
| dose 0, h=8 | supported |
| dose 5, h=1 | supported |
| dose 5, h=8 | outside empirical support |

`SupportReport.status` is `extrapolative` (partially extrapolative). The union
of the two horizon ranges is approximately `[-0.05, 10]` and would have labelled
dose 5 supported on the whole surface.

This fixture pins support geometry, not a structural mean surface. The
extrapolated cell's numerical value is not a known-truth claim.
