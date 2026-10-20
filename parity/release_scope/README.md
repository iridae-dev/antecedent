# Release scope (B05)

`scope.toml` is schema 1. Checking that file records the mandatory list and the research list. Later attempts write the claim they earned, or the adjusted claim after the named attempts are exhausted, back into the matching row. An unresolved mandatory row blocks release. A public claim changes only after those named attempts are exhausted. Changing either list takes an explicit reviewed release decision and the matching claim change.

`parity/method_specs/specs.toml` remains the record of the failed Gaussian-panel attempt. The four X02 specs there are `not_admitted`. `X02.gaussian_panel` in this file only freezes that attempt's release status.

## How to read a row

Each `[[entry]]` has an `id`, a `status`, and a `list`.

| status | Meaning |
| --- | --- |
| `mandatory` | Required for release. `obligation = "fit"` is the finite-lag family that must still be fit. `obligation = "flagship"` is X06. |
| `pending_attempt` | The next attempt, in `attempt_order`. `end_state` is the one-line result that attempt is aiming at. Unresolved until an outcome is written back. |
| `closed_failed` | An attempt that already failed. `attempt_order = 0` is the Gaussian panel B04 closed. |
| `research` | Not a release blocker. |

`list = "mandatory"` includes mandatory rows, pending attempts, and the closed attempt. `list = "research"` does not block release.

`source` says where the id came from: `route_manifest`, `method_children`, `feedback`, `workflow`, `joint_response`, `attempt`, or `research`.

`delta_ab_identified_claim = "pending"` until `X02.supported_worlds` and `X02.joint_assignment` have outcomes. A world becomes model-conditional only after both of those designs leave it unsupported.

## What is included

Mandatory rows are the licensed routes in `parity/capabilities/route_manifest.toml` (`status = "licensed"`), the B03 counterparts whose gap is not a closed research boundary, the feedback minimum (Y01 cyclic graph, Y02 SCC schedule, Y03 linear solution contract, Y10 experimental recovery recipe, Y09 sigma-separation), Q04, X01's two-intervention contrast on `licensed.AverageEffect.Dag`, the four X02 fit rows, and the two X06 flagship rows.

A closed research boundary is not a row. That is a B03 child whose B04 failure policy starts with `Refuse this axis`, or whose route is already `closed.*` and whose failure policy starts with `Identification is blocked`. X03, X04, and X05 are the pending attempt rows rather than a second copy of those children.

The research list is the four `X02.broader` children and G08. The 20 October 2026 scope amendment promotes Y18, Y19 and their ten acceptance children to mandatory 3.0 implementation work. Calibration and release attestation remain separate from those implementation tickets. The executing acceptance contract is `parity/cyclic/acceptance.toml`.

Graphless, closed, `execution_only`, and `fitting_gap` manifest rows are not licensed routes in this file.

## Counts by status

| status | Rows |
| --- | --- |
| mandatory | 204 |
| pending_attempt | 10 |
| research | 5 |
| closed_failed | 1 |

204 mandatory rows: the original 192 rows (91 licensed routes, 94 counterparts, 5 feedback rows, Q04, and `X01.licensed_contrast`), plus Y18/Y19 and their ten acceptance children.

For Y18/Y19 and their acceptance children, `implementation = "complete"` records the passed non-calibration implementation gate; `implementation_evidence` links the executing contract. Their `mandatory` status retains release scope. Calibration and attestation remain separate and unclaimed.
