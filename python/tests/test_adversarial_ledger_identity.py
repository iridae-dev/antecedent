"""Generated cases: evidence-ledger identities under reordering and under relevant changes.

The identity rules make a pooling's sources order-insensitive, and the ledger holds a
transport's source populations and assumptions as sets. A permitted reordering, or naming
a population or assumption twice, must keep the identity; adding a new declaration,
dropping one, changing the target, or moving one between the two lists must change it. The expected outcome comes from those rules (proved for the
canonical encoding in formal/Antecedent/Identity.lean), never from an earlier output.
"""

from __future__ import annotations

from antecedent import evidence_ledger as el
from antecedent.outcomes import require_success
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)

labels = st.text(alphabet="abcdefgh_", min_size=1, max_size=6)


def declaration(index: int) -> el.SourceDeclaration:
    return el.SourceDeclaration(
        f"study_{index}",
        "observation_set",
        observations=el.rows("cohort", range(index * 10, index * 10 + 10)),
    )


def pooled(order: list[int], count: int) -> str:
    """Result identity of pooling `count` disjoint studies, summaries given in `order`."""
    ledger = el.EvidenceLedger()
    ids = [ledger.add_source(declaration(i)) for i in range(count)]
    summaries = [el.StudySummary(ids[i], float(i), 1.0 + i) for i in order]
    return require_success(ledger.pool_summaries("pooled", summaries)).result


@SETTINGS
@given(
    st.integers(min_value=2, max_value=5).flatmap(
        lambda n: st.tuples(st.just(n), st.permutations(list(range(n))))
    )
)
def test_reordering_pooled_sources_keeps_the_result_identity(case):
    count, order = case
    assert pooled(order, count) == pooled(list(range(count)), count)


@SETTINGS
@given(st.integers(min_value=3, max_value=5), st.data())
def test_dropping_a_pooled_source_changes_the_result_identity(count, data):
    dropped = data.draw(st.integers(min_value=0, max_value=count - 1))
    kept = [i for i in range(count) if i != dropped]
    ledger = el.EvidenceLedger()
    ids = [ledger.add_source(declaration(i)) for i in range(count)]
    everything = require_success(
        ledger.pool_summaries(
            "pooled", [el.StudySummary(ids[i], float(i), 1.0 + i) for i in range(count)]
        )
    ).result
    fewer = require_success(
        ledger.pool_summaries("pooled", [el.StudySummary(ids[i], float(i), 1.0 + i) for i in kept])
    ).result
    assert everything != fewer


def transported_source(populations, assumptions, target="target") -> str:
    ledger = el.EvidenceLedger()
    return ledger.add_source(
        el.SourceDeclaration(
            "transported",
            "external_claim",
            observations=el.rows("trials", range(100, 110)),
            transport=el.TransportIdentity(
                source_populations=tuple(populations),
                target_population=target,
                selection_diagram=bytes([7] * 32).hex(),
                assumptions=tuple(assumptions),
            ),
        )
    )


@SETTINGS
@given(
    st.lists(labels, min_size=1, max_size=4).flatmap(
        lambda xs: st.tuples(st.just(xs), st.permutations(xs))
    ),
    st.lists(labels, min_size=1, max_size=4).flatmap(
        lambda xs: st.tuples(st.just(xs), st.permutations(xs))
    ),
)
def test_reordering_transport_declarations_keeps_the_source_identity(populations, assumptions):
    (pops, pops_reordered), (assumed, assumed_reordered) = populations, assumptions
    assert transported_source(pops, assumed) == transported_source(
        pops_reordered, assumed_reordered
    )


@SETTINGS
@given(
    st.lists(labels, min_size=1, max_size=4, unique=True),
    st.lists(labels, min_size=1, max_size=4, unique=True),
    labels,
)
def test_a_relevant_transport_change_changes_the_source_identity(populations, assumptions, extra):
    base = transported_source(populations, assumptions)
    if extra not in populations:
        assert transported_source([*populations, extra], assumptions) != base
    if extra not in assumptions:
        assert transported_source(populations, [*assumptions, extra]) != base
    assert transported_source(populations, assumptions, target="elsewhere") != base
    # The two lists cannot trade an element: moving one changes the identity.
    if len(populations) > 1 and populations[0] not in assumptions:
        moved = transported_source(populations[1:], [*assumptions, populations[0]])
        assert moved != base


@SETTINGS
@given(
    st.lists(labels, min_size=1, max_size=4), st.lists(labels, min_size=1, max_size=4), st.data()
)
def test_repeating_a_transport_declaration_keeps_the_source_identity(
    populations, assumptions, data
):
    """The ledger holds a transport's populations and assumptions as sets: naming one twice
    declares nothing new, so the identity is unchanged."""
    repeated_population = data.draw(st.sampled_from(populations))
    repeated_assumption = data.draw(st.sampled_from(assumptions))
    base = transported_source(populations, assumptions)
    assert transported_source([*populations, repeated_population], assumptions) == base
    assert transported_source(populations, [*assumptions, repeated_assumption]) == base
