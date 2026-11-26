# Use your own outcome and treatment models

You have a model that predicts the outcome, and one that predicts who is treated. You want the average effect of the treatment, with those models making the predictions. Antecedent still decides the question, the adjustment set, and which rows each fit may see.

The example uses a simulated table of 200 people. Each row has a covariate `z`. Treatment is assigned independently of `z`, to half the rows. The outcome is `1 + 2 * treatment`, so the average effect in the table is 2.

You declare that `z` may affect both treatment and the outcome, and that treatment may affect the outcome. Given `z`, that is the assumption of no unmeasured confounding. The simulation gives `z` neither role, and nothing below tests the assumption. Both pairs of models still recover 2: under this graph the adjustment is valid, and in this table `z` does not confound treatment.

Run `python examples/python/external_learner.py`. See the [complete example](#complete-runnable-source).

## Build a new model for every fit

Pass a function that returns a new, unfitted model. `learners.regressor` wraps the outcome model, which implements `fit` and `predict`. `learners.classifier` wraps the propensity model, which implements `fit` and `predict_proba`. The probability of class 1 is the chance of treatment. `Learners` holds the pair, and `analyze` takes that pair with `estimator="aipw"` or `estimator="dml"`.

```python
from antecedent import learners
from antecedent.external_learners import Learners

models = Learners(
    outcome=learners.regressor(lambda: LinearRegression()),
    treatment=learners.classifier(lambda: LogisticRegression()),
)
```

`LinearRegression` and `LogisticRegression` are the scikit-learn names. The same call accepts any class with those methods, including `lambda: NoriRegressor(...)` or `lambda: KumoClassifier(...)`. The example imports none of those packages. It defines a small least-squares class and a small logistic class, then repeats the analysis with an in-context mean.

Antecedent calls your function once while reading the declaration, and again for every fold. Each fold therefore trains a model that has not seen another fold. Put the model's settings inside the function. Leave the outcome column out of it. The rows of a fold arrive later, as the arguments of `fit`.

Define the class where Python can import it by name. Antecedent keeps the fitted model and loads that same object back when it needs a prediction.

```python
models = Learners(
    outcome=learners.regressor(LeastSquares, configuration={"intercept": True}),
    treatment=learners.classifier(LogisticProbability, seed=7),
)
outcome = ant.analyze(
    data,
    graph=[("z", "treatment"), ("z", "outcome"), ("treatment", "outcome")],
    query=ant.AverageEffect("treatment", "outcome"),
    learners=models,
    estimator="aipw",
)
```

`configuration` is stored with the study. Two studies that differ only in that mapping compile to different programs. When the model has a `random_state` parameter, `seed` is applied to each new model before `fit`. When `fit` accepts `sample_weight`, the row weights are passed through.

## An in-context model is the same shape

Some models answer from the rows in front of them rather than from a coefficient vector. For one fold, those rows are the ones `fit` receives. Store them there. `predict` then answers from what that call stored.

```python
class InContextMean:
    def fit(self, x, y, sample_weight=None):
        self._mean = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, x):
        return np.full(len(x), self._mean)
```

This class keeps the mean of its own fit. A row from another fold is not there to be read. A function that closed over the whole outcome column would carry held-out labels into every fold.

## What the 2 is

The example prints `2.000000` for the least-squares and logistic pair, and again for the in-context mean. Each figure is the average effect of treatment on the outcome under the graph above: the outcome if everyone were treated, minus the outcome if no one were.

The result names your factory, the configuration, and the request each fit received. Training that the model did inside `fit` stays on your side of that record. Agreement with 2 checks the calculation against the equation that generated the outcome. `z` does not influence treatment in this table, so the adjustment has nothing to remove, and the match does not test the graph.

Prediction uses the fitted object. A route that has to predict from saved bytes alone, with no live model, refuses the pair. Smoothed dose, trial AIPW, and joint-cell scores are that kind of route. This average-effect fit trains and predicts in one process, so it runs.

## What this established

On the simulated table, both pairs give an average effect of 2. Each fold trained a new model. The in-context mean used only the rows of its own fit.

## What this did not establish

The 2 is the effect written into the simulation. It is not a finding about a treatment anyone received. The graph remains an assumption. A model with `fit` and `predict` has the shape this calculation asks for, and that shape says nothing about whether the model is the right one for another question.

## Complete runnable source

<!-- example: examples/python/external_learner.py -->

[Open the Python script](../../examples/python/external_learner.py)
