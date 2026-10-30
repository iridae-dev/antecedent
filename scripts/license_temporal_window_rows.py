#!/usr/bin/env python3
"""Write the licensed rows of the cells that read the unrolled window of a stationary template.

Four blocks: the pulse, sustained and conditional contrasts on a supplied or accepted
TemporalAdmg (`window_contrasts`) and over a graph posterior of lagged TemporalAdmg atoms
(`window_contrast_posterior`), and the mediation decomposition on a supplied or accepted
TemporalAdmg (`window_mediation`), over a graph posterior of lagged TemporalAdmg atoms
(`window_mediation_posterior`) and on a TemporalPag (`window_mediation_pag`). The rows of a block
share one fixture and one evidence test file; only the query, structure source, inference mode and
validation differ. Each block sits between marker comments at the end of
parity/support_licensed.toml and is rewritten in place, so the script is idempotent:

    python3 scripts/license_temporal_window_rows.py          # rewrite the blocks
    python3 scripts/license_temporal_window_rows.py --check  # fail if a block is stale

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSED = ROOT / "parity" / "support_licensed.toml"
SELF = "scripts/license_temporal_window_rows.py"

FIXTURE = "conformance/estimate/temporal_window_truth"
TEST = "crates/antecedent/tests/temporal_window_route_evidence.rs"
ESTIMATOR = "functional.effect"

INFERENCES = ("Frequentist", "Bayesian")
ALL_VALIDATIONS = ("none", "cheap", "full")
CONTRAST_QUERIES = (
    "PulseEffect",
    "SustainedEffect",
    "TemporalConditionalPulse",
    "TemporalConditionalSustained",
)

CONTRAST_TEXT = {
    "PulseEffect": (
        "a pulse of the treatment at the first treated period read at the next period",
        "pulse 1/8",
    ),
    "SustainedEffect": (
        "the treatment held at the first treated period and the one after it, read at the "
        "second",
        "sustained 3/8",
    ),
    "TemporalConditionalPulse": (
        "a pulse of the treatment contrasted within the levels of a modifier read at the first "
        "treated period and averaged over the modifier",
        "conditional pulse 1/8",
    ),
    "TemporalConditionalSustained": (
        "the held treatment contrasted within the levels of a modifier read at the first treated "
        "period and averaged over the modifier",
        "conditional sustained 3/8",
    ),
}

# What each inference does with the same functional.
INFERENCE_TEXT = {
    "Frequentist": (
        "Frequentist: the identified functional is evaluated by the finite-discrete plug-in on "
        "the lag-aligned window table, with circular blocks of consecutive windows (never shorter "
        "than the window) for uncertainty when replicates are requested."
    ),
    "Bayesian": (
        "Bayesian: one Dirichlet weight is shared by each block of consecutive windows and the "
        "posterior mean is checked within the fixture's declared band."
    ),
}

CONTRAST_VALIDATION = (
    "Cheap validation refits the plug-in on a random contiguous stretch of 80% of the windows "
    "and on windows whose treatment columns are rotated together by whole windows; full adds a "
    "50% stretch and a split-half stationarity test. On the exact-law series the stretch refits "
    "return the contrast within 0.1, the rotated treatment gives a contrast within 0.1 of zero "
    "and each half carries the enumerated contrast; on the series whose treatment is reversed "
    "halfway (each half the exact law of its own process, contrasts negated) the split-half "
    "check fails with p below 0.01 while the pooled contrast cancels. The verdicts of the clean "
    "series are not asserted: the stretch, placebo and split-half p-values are diagnostics at "
    "a 5% level, not a calibration claim."
)

MEDIATION_VALIDATION = (
    "Cheap and full validation run the mediation-native suite on the window table. On the "
    "exact-law series the mediator rotated by whole windows keeps the total effect and removes "
    "the indirect one (within 0.05) and every subset refit returns the enumerated component "
    "(within 0.05); on a TemporalPag the placebo reports of every member of the identified set "
    "bracket the enumerated indirect effect on the clean series and report zero before and after "
    "rotation on a series whose mediator is detached from the outcome; the sensitivity grid is a diagnostic the reference does not assert, and the "
    "verdicts are checks, not a calibration claim."
)


def validation_text(validation: str, family: str) -> str:
    if validation == "none":
        return "Cheap and full validation are separate cells. "
    suite = CONTRAST_VALIDATION if family == "contrast" else MEDIATION_VALIDATION
    return f"Validation {validation}: " + suite + " "


def contrast_limitations(query: str, structure: str, inference: str, validation: str) -> str:
    what, value = CONTRAST_TEXT[query]
    posterior = structure == "graph_posterior"
    source = (
        "a graph posterior of lagged TemporalAdmg atoms, each atom identified on the unrolled "
        "window of its own template and mixed by posterior weight"
        if posterior
        else "a supplied or accepted TemporalAdmg"
    )
    mixture = (
        "The fixture posterior has three atoms: the generating template (weight 0.5, the "
        "enumerated value), a template without the treatment edge (weight 0.3, exactly zero) "
        "and a bow-arc template (weight 0.2, a hedge, unidentified mass): the mixture is the "
        "posterior-weighted mean over the identified atoms, 0.625 of the enumerated value, the "
        "unidentified mass is 0.2 and the status resolves to graph-dependent. "
        if posterior
        else "The fixture also reads a template that declares a latent cause of the "
        "covariate and the outcome, which the district formula identifies to the same value. "
    )
    return (
        f"Contrast of {what}, on {source}, unconditional on any variable but the modifier, "
        f"finite-discrete series. The window holds three periods (the history, the treated "
        f"period and the outcome period) and is identified by the complete ID algorithm "
        f"(IDC for a conditional contrast). The enumerated truth of the fixture is {value}. "
        f"{INFERENCE_TEXT[inference]} {mixture}"
        "The contrast equals the exact enumerated truth of the generating structural model on "
        "a series whose consecutive windows have exactly the law of the process "
        "(conformance/estimate/temporal_window_truth; the independent generator evaluates the "
        "structural equations under the intervention on every noise cell, not an "
        "identification formula). A window that holds a hedge is refused with the next "
        "attempts, and a modifier a treated period can reach is refused. "
        + validation_text(validation, "contrast")
        + "Contemporaneous atoms and multiple outcome horizons are refused. No sampling "
        "interval is claimed to cover; no coverage record measures this construction."
    )


def mediation_limitations(
    structure: str, inference: str, validation: str, klass: str
) -> str:
    if klass == "TemporalPag":
        source = (
            "a graph posterior of TemporalPag atoms"
            if structure == "graph_posterior"
            else "a supplied or accepted TemporalPag"
        )
        reading = (
            "each stationary MAG completion of the template is unfolded one template lag "
            "further back than the window, marginalized onto it by the inducing-path rule and "
            "read through the ADMGs that share its MAG; the contrast is reported only when "
            "every candidate identifies it, as a point when the candidates yield one functional "
            "and as the identified set over the distinct functionals otherwise. The fixture "
            "checks that the enumerated total, natural direct and pure natural indirect effects "
            "lie in the published identified set (or equal the point). "
        )
        extra = (
            "The fixture posterior has two atoms: the template with a parent of the treatment "
            "(weight 0.7) and the template with none, whose edges are invisible and whose "
            "mediation is not identified (weight 0.3, unidentified mass); the status resolves "
            "to graph-dependent. "
            if structure == "graph_posterior"
            else "A treatment with only invisible edges is refused as not identified. "
        )
    else:
        source = (
            "a graph posterior of lagged TemporalAdmg atoms, each atom identified on the "
            "unrolled window of its own template and mixed by posterior weight"
            if structure == "graph_posterior"
            else "a supplied or accepted TemporalAdmg"
        )
        reading = (
            "the path-specific identifier decides the total, natural direct and pure natural "
            "indirect effects on the unrolled window ADMG (the edge g-formula, or the district "
            "formula with the recanting-district criterion). "
        )
        extra = (
            "The fixture posterior has three atoms: the generating template (weight 0.5), a "
            "template whose mediator does not reach the outcome (weight 0.3, no path runs "
            "through the mediator, so the indirect effect is not identified there: unidentified "
            "mass) and a template with a latent cause of the mediator and the outcome one period "
            "earlier (weight 0.2, identified by the district formula to the same value): the "
            "indirect effect is the posterior-weighted mean over the identified atoms, the "
            "unidentified mass is 0.3 and the status resolves to graph-dependent. "
            if structure == "graph_posterior"
            else "A latent common cause of the mediator and the outcome in the same period "
            "is refused as not identified. "
        )
    return (
        f"Natural mediation of one treatment on one outcome at the pulse period through the "
        f"declared mediator, on {source}, finite-discrete series: {reading}"
        "The treatment is the pulse period's coordinate, the outcome the horizon period's "
        "coordinate and the mediators the declared variable at every period between them. "
        f"{INFERENCE_TEXT[inference]} {extra}"
        "The total effect 5/16, the natural direct effect 5/16 and the pure natural indirect "
        "effect 1/8 equal the exact enumerated truths of the generating structural model on a "
        "series whose consecutive windows have exactly the law of the process "
        "(conformance/estimate/temporal_window_truth; the independent generator evaluates the "
        "outcome equation under each treatment and mediator-treatment argument, not a "
        "mediation formula); the pure indirect effect is never total minus direct. "
        + validation_text(validation, "mediation")
        + "One outcome horizon per study. No sampling interval is claimed to cover; no "
        "coverage record measures this construction."
    )


def row(
    query: str,
    klass: str,
    structure: str,
    inference: str,
    validation: str,
    assertion: str,
    checked: str,
    limitations: str,
) -> str:
    return (
        "[[cell]]\n"
        f'query = "{query}"\n'
        f'graph_class = "{klass}"\n'
        f'structure = "{structure}"\n'
        f'inference = "{inference}"\n'
        f'validation = "{validation}"\n'
        f'estimators = ["{ESTIMATOR}"]\n'
        "staged = true\n"
        'evidence_kind = "internal_known_truth"\n'
        f'known_truth_fixture = "{FIXTURE}"\n'
        f'evidence_test = "{TEST}"\n'
        f'evidence_assertion = "{assertion}"\n'
        "checked_execution = [\n"
        f'  {{ estimator = "{ESTIMATOR}", test = "{TEST}", assertion = "{checked}" }},\n'
        "]\n"
        f'limitations = "{limitations}"\n'
        'calibration_reason = "estimator_grid_not_measured"\n'
    )


def block(queries, klass, structures, validations, assertions, checked, limitations) -> str:
    out = []
    for query in queries:
        for structure in structures:
            for inference in INFERENCES:
                for validation in validations:
                    out.append(
                        row(
                            query,
                            klass,
                            structure,
                            inference,
                            validation,
                            assertions[validation == "none"],
                            checked,
                            limitations(query, structure, inference, validation),
                        )
                    )
    return "\n".join(out)


def window_contrasts() -> str:
    return block(
        CONTRAST_QUERIES,
        "TemporalAdmg",
        ("explicit", "accepted"),
        ALL_VALIDATIONS,
        (
            "temporal_admg_window_validation_reports_the_known_refits_and_detects_a_planted_break",
            "temporal_admg_window_contrasts_match_the_enumerated_truth_on_both_inferences",
        ),
        "temporal_admg_window_contrasts_run_from_the_retained_plan_without_the_builder",
        contrast_limitations,
    )


def window_contrast_posterior() -> str:
    return block(
        CONTRAST_QUERIES,
        "TemporalAdmg",
        ("graph_posterior",),
        ALL_VALIDATIONS,
        (
            "temporal_admg_window_posterior_validation_mixes_the_per_atom_window_suite",
            "temporal_admg_window_posterior_mixes_by_posterior_weight_and_accounts_for_unidentified_mass",
        ),
        "temporal_admg_window_posterior_runs_from_the_retained_plan_without_the_builder",
        contrast_limitations,
    )


def window_mediation() -> str:
    return block(
        ("TemporalMediationEffect",),
        "TemporalAdmg",
        ("explicit", "accepted"),
        ALL_VALIDATIONS,
        (
            "temporal_admg_window_mediation_validation_runs_the_mediation_suite_on_the_window_table",
            "temporal_admg_window_mediation_matches_the_enumerated_decomposition_on_both_inferences",
        ),
        "temporal_admg_window_mediation_runs_from_the_retained_plan_without_the_builder",
        lambda _query, structure, inference, validation: mediation_limitations(
            structure, inference, validation, "TemporalAdmg"
        ),
    )


def window_mediation_posterior() -> str:
    return block(
        ("TemporalMediationEffect",),
        "TemporalAdmg",
        ("graph_posterior",),
        ALL_VALIDATIONS,
        (
            "temporal_admg_window_mediation_posterior_validation_mixes_the_per_atom_reports",
            "temporal_admg_window_mediation_posterior_mixes_each_component_over_the_atoms_that_identify_it",
        ),
        "temporal_admg_window_mediation_posterior_mixes_each_component_over_the_atoms_that_identify_it",
        lambda _query, structure, inference, validation: mediation_limitations(
            structure, inference, validation, "TemporalAdmg"
        ),
    )


def window_mediation_pag() -> str:
    validation = "temporal_pag_window_mediation_validation_runs_the_mediation_suite_on_every_member"

    def text(_query, structure, inference, validation_axis):
        return mediation_limitations(structure, inference, validation_axis, "TemporalPag")

    explicit = block(
        ("TemporalMediationEffect",),
        "TemporalPag",
        ("explicit", "accepted"),
        ALL_VALIDATIONS,
        (
            validation,
            "temporal_pag_window_mediation_brackets_the_enumerated_decomposition_on_both_inferences",
        ),
        "temporal_pag_window_mediation_runs_from_the_retained_plan_without_the_builder",
        text,
    )
    posterior = block(
        ("TemporalMediationEffect",),
        "TemporalPag",
        ("graph_posterior",),
        ALL_VALIDATIONS,
        (
            validation,
            "temporal_pag_window_mediation_posterior_accounts_for_the_atom_it_cannot_identify",
        ),
        "temporal_pag_window_mediation_posterior_accounts_for_the_atom_it_cannot_identify",
        text,
    )
    return explicit + "\n" + posterior


BLOCKS = {
    "window_contrasts": window_contrasts,
    "window_contrast_posterior": window_contrast_posterior,
    "window_mediation": window_mediation,
    "window_mediation_posterior": window_mediation_posterior,
    "window_mediation_pag": window_mediation_pag,
}


def markers(name: str) -> tuple[str, str]:
    return (
        f"# generated:{name}:begin ({SELF})\n",
        f"# generated:{name}:end\n",
    )


def apply(text: str) -> str:
    for name, rows in BLOCKS.items():
        begin, end = markers(name)
        fresh = begin + "\n" + rows() + "\n" + end
        if begin in text:
            head, rest = text.split(begin, 1)
            _, tail = rest.split(end, 1)
            text = head + fresh + tail
        else:
            text = text.rstrip("\n") + "\n\n" + fresh
    return text


def main() -> int:
    text = LICENSED.read_text(encoding="utf-8")
    updated = apply(text)
    if "--check" in sys.argv:
        if updated != text:
            print("parity/support_licensed.toml: a temporal window block is stale")
            return 1
        return 0
    if updated != text:
        LICENSED.write_text(updated, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
