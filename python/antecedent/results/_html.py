"""Self-contained ``_repr_html_`` rendering for notebook display.

Jupyter calls ``_repr_html_`` on the last expression of a cell. Left bare,
:class:`~antecedent.results.AnalysisResult` printed the default dataclass
repr — a wall of nested field names — in the three README-linked Colab
notebooks, which are the top of the funnel for anyone evaluating this
library. This module renders a compact verdict banner, effect summary,
adjustment-set chips, and refutation table instead, and attaches the
renderer to :class:`AnalysisResult` plus the two nested views worth
displaying standalone (:class:`ValidationView`, :class:`PosteriorView`).

Every interpolated value — including numbers already formatted to strings —
goes through :func:`html.escape` with ``quote=True``. Variable names and
estimator/refuter ids originate in caller data or native strings and can
contain ``<``, ``&``, or quotes; nothing here trusts them. The markup is a
single inline ``<style>`` block with no external assets, and it avoids
committing to a light or dark background so it stays legible under either
notebook theme. Rendering never raises: any failure falls back to
``repr(self)`` in a ``<pre>`` block rather than breaking the notebook.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from ._format import fmt_float, fmt_pct

if TYPE_CHECKING:
    from ._views import AnalysisResult, PosteriorView, ValidationView

__all__: list[str] = []


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


_STYLE = """<style>
.antecedent-ar-card {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  border: 1px solid rgba(128, 128, 128, 0.35);
  border-radius: 8px;
  padding: 12px 16px;
  max-width: 680px;
  color: inherit;
  background: rgba(128, 128, 128, 0.06);
  line-height: 1.4;
}
.antecedent-ar-banner {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 6px;
  margin-bottom: 8px;
  font-weight: 600;
}
.antecedent-ar-banner.ar-ok { background: rgba(16, 185, 129, 0.18); }
.antecedent-ar-banner.ar-bad { background: rgba(239, 68, 68, 0.18); }
.antecedent-ar-banner-method { font-weight: 400; opacity: 0.75; font-size: 0.85em; }
.antecedent-ar-callout {
  background: rgba(245, 158, 11, 0.20);
  border-left: 4px solid #f59e0b;
  padding: 8px 10px;
  border-radius: 4px;
  margin-bottom: 10px;
  font-size: 0.92em;
}
.antecedent-ar-row {
  margin: 6px 0;
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 8px;
}
.antecedent-ar-label { font-weight: 600; min-width: 110px; opacity: 0.85; }
.antecedent-ar-value { font-variant-numeric: tabular-nums; }
.antecedent-ar-sub { opacity: 0.7; font-size: 0.85em; }
.antecedent-ar-muted { opacity: 0.6; font-style: italic; }
.antecedent-ar-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.antecedent-ar-chip {
  background: rgba(59, 130, 246, 0.18);
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 0.85em;
}
.antecedent-ar-table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 0.9em; }
.antecedent-ar-table th, .antecedent-ar-table td {
  border-bottom: 1px solid rgba(128, 128, 128, 0.25);
  padding: 4px 8px;
  text-align: left;
}
.antecedent-ar-table td.ar-ok { color: #10b981; font-weight: 600; }
.antecedent-ar-table td.ar-bad { color: #ef4444; font-weight: 600; }
</style>"""


def _unidentified_callout_html(mass: float | None) -> str:
    """Amber callout — the single most important element when it fires."""
    if mass is None or not (mass > 0):
        return ""
    pct = _esc(fmt_pct(mass))
    return (
        '<div class="antecedent-ar-callout">'
        f"<strong>{pct} of the graph posterior gives no identified estimand.</strong> "
        "The effect below only averages over the structures where it is identified; "
        "the rest of the posterior says the data alone cannot answer this question "
        "under those graphs."
        "</div>"
    )


def _refutation_table_html(validation: ValidationView) -> str:
    if not validation.ran or len(validation) == 0:
        return '<div class="antecedent-ar-row antecedent-ar-muted">No refutations ran.</div>'
    rows = []
    for r in validation.reports:
        verdict = "pass" if r.passed else "fail"
        verdict_class = "ar-ok" if r.passed else "ar-bad"
        rows.append(
            "<tr>"
            f"<td>{_esc(r.refuter)}</td>"
            f"<td>{_esc(fmt_float(r.original_ate))}</td>"
            f"<td>{_esc(fmt_float(r.refuted_ate))}</td>"
            f"<td>{_esc(fmt_float(r.comparison))}</td>"
            f'<td class="{verdict_class}">{_esc(verdict)}</td>'
            "</tr>"
        )
    body = "".join(rows)
    return (
        '<table class="antecedent-ar-table">'
        "<thead><tr>"
        "<th>Refuter</th><th>Original</th><th>Refuted</th><th>Comparison</th><th>Result</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def _adjustment_chips_html(adjustment_set: list[str]) -> str:
    if not adjustment_set:
        return '<span class="antecedent-ar-muted">(none)</span>'
    chips = "".join(f'<span class="antecedent-ar-chip">{_esc(v)}</span>' for v in adjustment_set)
    return f'<span class="antecedent-ar-chips">{chips}</span>'


def _analysis_result_body(result: AnalysisResult) -> str:
    ident = result.identification
    identified = bool(ident)
    banner_class = "ar-ok" if identified else "ar-bad"
    banner_text = "Identified" if identified else "Not identified"

    se = (
        result.estimate.se_bootstrap
        if result.estimate.se_bootstrap is not None
        else result.estimate.se_analytic
    )
    se_kind = "bootstrap" if result.estimate.se_bootstrap is not None else "analytic"

    mass = result.posterior.unidentified_mass if result.posterior is not None else None
    callout = _unidentified_callout_html(mass)
    refute_table = _refutation_table_html(result.validation)
    chips = _adjustment_chips_html(ident.adjustment_set)

    return (
        f'<div class="antecedent-ar-card">'
        f'<div class="antecedent-ar-banner {banner_class}">'
        f"<span>{_esc(banner_text)}</span>"
        f'<span class="antecedent-ar-banner-method">via {_esc(ident.method)}</span>'
        f"</div>"
        f"{callout}"
        f'<div class="antecedent-ar-row">'
        f'<span class="antecedent-ar-label">Effect</span>'
        f'<span class="antecedent-ar-value">'
        f"{_esc(fmt_float(result.effect))} ± {_esc(fmt_float(se))} ({_esc(se_kind)})"
        f"</span>"
        f'<span class="antecedent-ar-sub">estimator: {_esc(result.estimate.estimator_id)}</span>'
        f"</div>"
        f'<div class="antecedent-ar-row">'
        f'<span class="antecedent-ar-label">Adjustment set</span>{chips}'
        f"</div>"
        f"{refute_table}"
        f"</div>"
    )


def _analysis_result_repr_html(self: AnalysisResult) -> str:
    try:
        return _STYLE + _analysis_result_body(self)
    except Exception:  # noqa: BLE001 — never break the notebook on display
        return f"<pre>{_esc(repr(self))}</pre>"


def _validation_repr_html(self: ValidationView) -> str:
    try:
        return _STYLE + f'<div class="antecedent-ar-card">{_refutation_table_html(self)}</div>'
    except Exception:  # noqa: BLE001 — never break the notebook on display
        return f"<pre>{_esc(repr(self))}</pre>"


def _posterior_repr_html(self: PosteriorView) -> str:
    try:
        if self.effect_mean is None:
            body = '<div class="antecedent-ar-muted">No posterior computed.</div>'
            return f'{_STYLE}<div class="antecedent-ar-card">{body}</div>'
        callout = _unidentified_callout_html(self.unidentified_mass)
        row = (
            '<div class="antecedent-ar-row">'
            '<span class="antecedent-ar-label">Posterior</span>'
            f'<span class="antecedent-ar-value">'
            f"{_esc(fmt_float(self.effect_mean))} ± {_esc(fmt_float(self.effect_sd))}"
            f"</span>"
            f'<span class="antecedent-ar-sub">'
            f"95% CI [{_esc(fmt_float(self.q025))}, {_esc(fmt_float(self.q975))}]"
            f"</span>"
            "</div>"
        )
        return f'{_STYLE}<div class="antecedent-ar-card">{callout}{row}</div>'
    except Exception:  # noqa: BLE001 — never break the notebook on display
        return f"<pre>{_esc(repr(self))}</pre>"


def _attach() -> None:
    """Attach ``_repr_html_`` to the view classes; called once on import."""
    from ._views import AnalysisResult, PosteriorView, ValidationView

    AnalysisResult._repr_html_ = _analysis_result_repr_html  # type: ignore[attr-defined]
    ValidationView._repr_html_ = _validation_repr_html  # type: ignore[attr-defined]
    PosteriorView._repr_html_ = _posterior_repr_html  # type: ignore[attr-defined]


_attach()
