# Readable report

A report turns a contract into a document a person can read: Markdown for a
repository, a review or a notebook, or one HTML page for a browser. It is
rendered from the records a study already holds, so it adds no analysis. Every
number in it is a value one of those records carries, printed to at most twelve
significant digits. The wording is fixed text chosen by the records' own status
tags, and the same records render to the same bytes: there is no clock, random
source, unordered iteration or environment read.

## Calls

| Where | Call | Reads |
| --- | --- | --- |
| Python | `prepared.report("markdown")` or `prepared.report("html")` | After an estimate, the executed claim and the contract it ran under; before one, the prepared contract. Always the plan view. |
| Rust, handle | `PreparedStudy::report(result, ctx, format)` | The same, with `result: Option<&StudyResult>`. |
| Rust, records | `antecedent::report::render_report(&ReportInput::new(&contract).with_claim(&claim).with_plan(&plan), format)` | Whatever records the caller holds. A contract from `Study::inspect()` alone reports a cell the support matrix refuses, with its blockers and next actions. |

Rendering identifies, fits and scans nothing, runs no estimate and changes no
handle. A Python handle that executes no lowered study program, such as a
transport stage, has no plan view and raises `plan_view_unavailable`. A cell
the support matrix refuses raises at `prepare`, so Python has no handle to
report on; the Rust contract-only report covers it.

## Sections

| Section | What it states |
| --- | --- |
| Summary | The query, inference, identifier, estimator, route, claim kind and value, the support status, and fixed sentences keyed to those statuses. |
| Claim | Only with a claim: claim id, kind, value and units, the identification, support and evaluated domains, and the execution digest. |
| Identification | Graph class, structure source, identifier and estimator, and the identification slot: status and the identified, unidentified, unevaluable and incomplete-search masses, or why the slot is unavailable. |
| Assumptions | The obligations with scope, kind, status, source, required check and description, and the assumptions the plan's identification relies on. |
| Diagnostics and evidence | The evidence the claim references, attested evidence (marked as not re-verifiable), checks that failed or have not run, and the obligations identification left unresolved. |
| Uncertainty | The uncertainty slot's components and whether each is omitted, the uncertainty the program must publish, and the calibration slot with its basis. |
| Support and licence | The support status, the support slot and the support cell of the program. |
| Refusals and limits | The operation report: applicability, readiness, blockers, unmet obligations, next actions and licensed neighbours, with the limits the other records state (a capped search, unidentified mass, omitted uncertainty, calibration that is not measured). |
| Program plan | The route, family, stage and identity, the scientific problem, the identification product and its estimands, the statistical, evidence and physical layers, and a graph posterior's mixture. The sized physical record and kernel policy describe the machine and are left out. |
| Identities | The domain-separated digests of the contract. |

A status is reported as the records carry it. A route that executes without a
support-matrix licence says so, and a calibration slot that is not `calibrated`
says that the reported uncertainty has no measured-coverage record. The report
grants no licence and upgrades no status.

## Escaping

The HTML is one self-contained document with inline style. It has no script
element, loads no resource and sets a content security policy that forbids
both. Every value taken from an input, such as a variable name, an obligation
description or an evidence reference, is HTML-escaped. The Markdown
backslash-escapes every markup character in the same values, and both formats
replace control characters with spaces so a value cannot start a new line or
row.

## Golden files

`crates/antecedent/tests/golden/report/` and `python/tests/golden/report/` pin
the output for a licensed static effect, a route that executes without a licence,
a Bayesian effect and a temporal effect, and (in Rust) a refused cell. Regenerate them
deliberately with `ANTECEDENT_REGENERATE_REPORTS=1` and review the diff.
