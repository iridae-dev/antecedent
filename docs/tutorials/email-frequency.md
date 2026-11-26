# Choose an email frequency when the cost of an unsubscribe is a range

The meeting goes the same way every quarter. Growth wants a fourth marketing email. Retention
has the unsubscribe chart. Finance will not put one number on a lost subscriber, and will not
let the argument proceed without one. The holdout already in the archive compared silence
with the cadence you send today. It never assigned the frequency in the middle, which is
where the margin flattens and the list starts to leave.

Two emails in a month still add margin. The fourth adds less of it, and costs more
subscribers. If a lost subscriber is worth 5 euros, the fourth email earns its place. If they
are worth 80, it does not. This page runs that argument on a simulated list whose answers are
known, so you can watch the estimate recover them, watch the choice turn at a price you can
check by hand, and see how the same argument continues once that price is on the table.

Run `python examples/python/email_frequency.py`. See the
[complete example](#the-complete-example).

## The list, and the fight over one number

Each customer is assigned 0, 2, or 4 emails with equal probability. The arm means are
declared, so a wrong coordinate cannot hide behind noise:

```text
margin       = 0, 1.20, 1.50 euros
unsubscribe  = 0.01, 0.02, 0.04
```

Silence earns nothing and loses almost nobody. Two emails bring 1.20 euros and lose 2
percent. Four emails bring only 30 cents more, and they double the unsubscribe rate again.
The noise inside each arm is a balanced set of normal quantiles, so the estimated mean lands
on that truth. A real list would scatter. The shape would still be the thing you came to
measure: margin that flattens, unsubscribes that speed up.

| You already know | The experiment estimates | You still have to judge |
| --- | --- | --- |
| The frequency was randomized, one of three actions per customer, with equal probability | Mean margin and mean unsubscribe rate under each frequency | What a lost subscriber is worth later: 5 to 80 euros |

Value per customer, in euros of this month plus the future you are arguing about:

```text
value = margin − unsubscribe_cost × unsubscribe
```

`unsubscribe_cost` is that argument, written down as a range instead of a pretended point.
Because margin flattens and unsubscribes speed up, the best frequency can change inside the
band finance will actually sign.

## The number on the result is the wrong number

Three frequencies are a multi-arm experiment. The fit reports a Horvitz–Thompson mean for
every arm, which is the level under each frequency, the thing the value reads. The number it
publishes as the result's point is something else: the second arm minus the first.

On margin that contrast is 1.20 euros. Silence earns nothing, so it happens to equal the mean
under two emails, and it is easy to think the result has handed you the level. On the
unsubscribe rate the disguise comes off. The contrast is 0.01. The rate under two emails is
0.02, and under four it is 0.04. A decision that uses 0.01 in place of those levels is
pricing the wrong list.

One analysis per outcome, both on the same assignment:

```python
def fitted(outcome):
    result = require_success(ant.analyze(data, query=RandomizedEffect(outcome, design)))
    fit = result.randomized_effect
    return result, {label: value for label, value, _variance, _count in fit.multi_arm_values}

results, means = {}, {}
for outcome in ("margin", "unsubscribe"):
    results[outcome], means[outcome] = fitted(outcome)
```

`means` is the level under each frequency: margin 0, 1.20, 1.50 and an unsubscribe rate of
0.01, 0.02, 0.04. `results["unsubscribe"].answer.value` is the contrast, 0.01. Each arm
carries a variance bound beside its mean. The value below asks for the expectation of a
straight trade of margin against unsubscribes, and the means answer it.

The contract asks for both outcomes at all three frequencies. Hand it the margin result alone
and it refuses. One result is one outcome, and its point is one contrast.

```python
try:
    require_success(contract.evaluate(results["margin"]))
except CausalValueError as refusal:
    refusal   # the contract declares two units; this result is one contrast
```

## The arm means are the decision

`decision.MultiArmEvidence` is how the levels reach the contract. You give it one result per
outcome, who the customers are, the units, and what each arm label means as an intervention.
The fit stores `"0"`, `"2"`, and `"4"`. You say that those are silence, two emails, and four.
Margin and the unsubscribe rate have to come from the same assignment. Estimate one of them
under a different assignment and the join refuses (`decision_adapters.invalid_claims`): two
outcomes of one decision share one list.

```python
evidence = decision.MultiArmEvidence(
    results=(results["margin"], results["unsubscribe"]),
    population="customers",
    units={"margin": "EUR", "unsubscribe": "unsubscribes"},
    regimes={"0": "do(emails=0)", "2": "do(emails=2)", "4": "do(emails=4)"},
)

def action(name, label):
    return decision.Action(
        name,
        inputs={
            "margin": decision.multi_arm_quantity(evidence, "margin", label),
            "unsubscribe": decision.multi_arm_quantity(evidence, "unsubscribe", label),
        },
    )
```

Each input is the mean of one arm, functional `mean`, at horizon 0.
`decision.multi_arm_quantity(evidence, "margin", "2")` is the 1.20 euros under two emails.

```python
margin = decision.attribute("margin", units="EUR")
unsubscribe = decision.attribute("unsubscribe", units="unsubscribes")
unsubscribe_cost = decision.tradeoff(
    "unsubscribe_cost",
    units="EUR/unsubscribe",
    lower=5.0,
    upper=80.0,
    assumption="customer_lifetime_value",
)
model = decision.Value(margin - unsubscribe_cost * unsubscribe, units="EUR")
contract = decision.Contract(
    value=model,
    actions=tuple(action(name, label) for name, label, _level in LEVELS),
    target_population="customers",
    policy=decision.TradeoffPolicy.invariant_best(),
)
decided = require_success(contract.evaluate(evidence))
```

`customer_lifetime_value` is the name of the judgment the band stands on, so a later reader
can see whose number it was. The value is a straight trade of the two means, so the exact
route answers from those means. A penalty that kicks in only after the unsubscribe rate
crosses a line is a different question: it needs the two outcomes together, as one law, and
these means are the answer to the question this meeting actually asked.

## Fifteen euros

Each frequency's value is a straight line in the cost of an unsubscribe, `w`:

| Send | Value per customer | As the cost runs from 5 to 80 euros | Most it can fall behind |
| --- | --- | --- | --- |
| none | −0.01w | −0.80 to −0.05 | 1.35 |
| two | 1.20 − 0.02w | −0.40 to 1.10 | 0.20 |
| four | 1.50 − 0.04w | −1.70 to 1.30 | 1.30 |

```python
decided.kind        # "preference_dependent"
decided.standing    # "depends"
decided.choice      # None
print(decided.statement)
```

No frequency wins at every cost finance will sign, so a policy that chooses only when one
send leads everywhere chooses nothing. That is the right silence. The evidence has done its
job. The remaining disagreement is the cost, and the statement names the band.

The value range is how far a send moves as that cost runs from 5 to 80. The last column is
the furthest that send falls behind the best alternative anywhere in the band. Two emails
never falls behind by more than 20 cents a customer. Four can fall behind by 1.30 euros, at
the expensive end of the band, where the extra unsubscribes have piled up. Silence can fall
behind by 1.35, at the cheap end, where four emails is collecting margin that silence never
sees.

Where the lines cross:

| Pair | Margin of the first over the second | They trade places at |
| --- | --- | --- |
| two, four | 0.02w − 0.30 | 15 euros per unsubscribe |
| none, four | 0.03w − 1.50 | 50 euros per unsubscribe |

```python
for switch in decided.switches:
    print(switch.first, switch.second, switch.threshold.value, switch.threshold.first_leads_above)
```

A switch names the two sends in the order you declared them. `first_leads_above` means the
first leads once the cost is past the threshold. Below 15 euros, send four. Above 15, send
two. Silence beats four only after an unsubscribe costs more than 50 euros, and by then two
emails beats both of them, so silence is never the send you want inside the band:
`possibly_best` is false for `none`. Silence and two emails would trade places at 120 euros,
outside anything finance put on the table, and that crossing is left out.

You can take one sentence back to the room. Four emails is the better send when a lost
subscriber is worth less than 15 euros, and two emails is the better send when they are worth
more. Someone in the room can answer that. The experiment has turned a fight about "more
email" into a price.

If the calendar will not wait for that answer, you are choosing a rule for living with the
band. `TradeoffPolicy.minimax_regret()` sends two emails, because 20 cents is the smallest
regret on the table. The standing of that pick is `policy_choice`: a rule you adopted, on top
of an experiment that already reported where the choice turns.

## The study in the archive, and the line someone will fit on Monday

The holdout the company already paid for assigned silence and four emails. It has no one who
received two. Compile this contract against that grid and the missing frequency stays
missing: `quantity_semantics_mismatch`, detail `program_binding.dose_grid_changed`. The
archive can support a comparison of the ends. It cannot tell you about the frequency where
the economics actually turn. That is why the experiment on this page assigns all three.

```python
studies = [prepared(data, outcome, [0.0, 2.0, 4.0], bayes=False) for outcome in ("margin", "unsubscribe")]
from_curves = require_success(require_success(contract.compile(studies)).execute())
```

Fit a mean response at those same three doses, with `emails` pointing at each outcome and
nothing else to adjust for, and the curve passes through 0, 1.20, and 1.50. The threshold
stays at 15 euros. On this list the response at the assigned doses and the Horvitz–Thompson
arm means agree, and the decision above is the decision of the arm means.

A conjugate linear dose model will also compile, and it will come to the meeting looking like
a summary. It draws a straight line through the arms. Margin on that line sits near 0.15,
0.90, and 1.65 euros: the flattening is gone, replaced by a steady 37 cents per extra email.
Both outcomes are straightened the same way, so every pair of frequencies switches near 50
euros, and silence comes back into contention. Bring that line to the room and you will
recommend a different send than the experiment supports, with confidence, at the wrong price.
The falloff lives in the arm means. A straight dose law smooths it out, and the choice moves
with it.

## Once the price is on the table

Fifteen euros answers the question the meeting brought. It also tells you which question is
worth asking next. Each of these is a real next analysis. Each has the design its own
question needs. They are how you keep going, and they are separate from the arm means above.

**Some people are already tired.** A fourth email to someone who opened the last three is a
different proposition from a fourth email to someone who has not opened anything since
January. A conditional dose response asks for the mean at each frequency inside a baseline
group you name. A policy asks what the list is worth if you mail the engaged and leave the
cold alone. Both are in
[heterogeneous effects and policies](../guides/heterogeneous-effects-and-policies.md). The
frequency decision above is the average customer. The policy question is who should have been
left out of the average.

**Another thousand customers, or send on what you have.** Once the choice turns at a known
price, more data is valuable only if it would change the send. Pricing that, net of the cost
of the test, is the calculation in
[Stop, continue or run longer](decide-under-uncertainty.md): expected value of sample
information, a ranking of candidate studies, and a comparison against deciding today. There
the unknown is an advertising effect with a standard error. Here the unknown you would be
buying down is whatever still sits between you and a confident side of 15 euros.

**The history was never an experiment.** If marketers already chose who received more mail,
because those people looked ready to buy, the arm means of that history mix the emails with
the people. A comparison then has to say what else differed. That is the shape of
[an effect from observational data](observational-study.md): the treatment was chosen, and
the estimate stands up only with the adjustment you are willing to defend. The randomized
list on this page is how you avoid having that argument. The archive of past campaigns is
how you end up in it.

**One inbox is shared.** A promo forwarded inside a household, or a roommate who marks the
domain as spam, means your customer's outcome depends on someone else's email. A plain arm
mean folds that into "the effect of frequency." Separating what your own send did from what
the surrounding sends did is a design with a network and an exposure, in
[interference and network exposure](../guides/designs/interference-and-network-exposure.md).
You bring the edges. The contrast is then between exposures, own mail and neighbors' mail,
rather than between 0, 2, and 4 as if every inbox were a closed room.

**Opens arrive in bursts.** A campaign lands, a few people forward it, and the next hour is
busy in a way the monthly mean does not describe. A Hawkes model estimates how often events
arrive on their own and how many follow-ups one event drags in. The worked story is
[when outside events interrupt a launch](event-driven-launch.md): fit the bursts, simulate
the next few hours, and ask whether acting on one more hour of observation would change the
call. Frequency over a month and attention over an hour are different clocks. The second
becomes the question when the complaint is no longer "how many emails" but "what happens in
the hour after you send."

**The next email depends on the last open.** A rule that waits, or discounts, or goes quiet
when the last message was ignored, is a policy over history. Writing that rule, replaying it
on the same trajectories, and comparing it with a fixed cadence is the work of
[write and replay causal policies](programmable-policies.md). The experiment on this page
assigns the frequency in advance and reads the mean. A cadence that reacts is the question
you ask after you know the average is not the decision you want to keep making by hand.

## What you can take into the meeting

At these means, with a lost subscriber worth anywhere from 5 to 80 euros, four emails leads
below 15 and two emails leads above it. Silence is never the best send inside that band. The
coordinates are the Horvitz–Thompson mean of each frequency, for margin and for the
unsubscribe rate, from one shared assignment. The holdout of the two ends cannot say this.
A straight line through the same points says something else, and it is the wrong send.

## What you still bring

The experiment does not pick the cost. You do, and you can say whose judgment the band was.
Past choices, recorded against the forecasts that were on the table at the time, can narrow a
band like this one. That is a different page:
[choose a discount when the cost of a return is a range](decide-across-trade-offs.md).

The ranges above move the cost, and the means stay the means you estimated. Putting a belief
on an uncertain effect, and asking what another study would be worth, is the advertising
tutorial linked above.

A charge that depends on the unsubscribe rate crossing a line needs the two outcomes as one
law. The expectation the meeting asked for, margin against a priced unsubscribe, is what
these means settle.

## Where to go next

The declarations, verdicts, and routes are in
[trade-off ranges in depth](../preference-decisions.md). The multi-arm estimator is in
[randomized experiments](../guides/designs/randomized-experiments.md). The same shape of
decision, with the cost of a return in the role of an unsubscribe, is the discount tutorial.
A decision with one exact utility is in [Decisions](../decisions.md).

## The complete example

<!-- example: examples/python/email_frequency.py -->

[Open the Python script](../../examples/python/email_frequency.py)
