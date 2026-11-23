"""Observed-data oracle for two-node signed-uniform Gaussian graph probabilities.

The first column is observed. The second can be missing completely at random.
The graph prior is uniform on the empty graph and the two directed graphs.
An edge has known unit residual variance and one signed-uniform coefficient
shared by every row. Return the three graph probabilities in that order.
"""

from math import exp, fsum, log, log1p


def log_uniform_integral(log_density, panels):
    """Composite midpoint quadrature on the two signed coefficient intervals."""
    width = 0.5 / panels
    values = [log_density(sign * (0.5 + (i + 0.5) * width))
              for sign in (-1, 1) for i in range(panels)]
    peak = max(values)
    return peak + log(width * fsum(exp(v - peak) for v in values))


def graph_probabilities(rows, panels=128):
    """Integrate each graph's likelihood relative to the independent unit normals."""
    complete = [(x, y) for x, y in rows if y is not None]
    missing = [x for x, y in rows if y is None]
    xy = fsum(x * y for x, y in complete)
    xx = fsum(x * x for x, _ in complete)
    yy = fsum(y * y for _, y in complete)
    missing_xx = fsum(x * x for x in missing)

    def reverse_log_ratio(b):
        # With the parent missing, X has marginal variance 1 + b*b.
        b2 = b * b
        log_ratio = b * xy - 0.5 * b2 * yy
        log_ratio += 0.5 * (missing_xx * b2 / (1 + b2) - len(missing) * log1p(b2))
        return log_ratio

    # When only the child is missing its conditional density integrates to one.
    log_weights = (0., log_uniform_integral(lambda b: b * xy - 0.5 * b * b * xx, panels),
                   log_uniform_integral(reverse_log_ratio, panels))
    peak = max(log_weights)
    weights = [exp(w - peak) for w in log_weights]
    return tuple(weight / fsum(weights) for weight in weights)


if __name__ == '__main__':
    # Verify by integrating an unobserved Y directly in the full two-variable
    # density and then integrating the shared coefficient. The observed cell is
    # bounded here, so the Gaussian tails beyond +/-12 are negligible.
    from math import isclose, pi, sqrt

    x = 1.25
    rows = [(0.8, -1.4), (-0.3, 0.1), (x, None)]
    observed = graph_probabilities(rows, panels=512)
    step = 0.05
    nodes = [(i + 0.5) * step for i in range(-240, 240)]
    def direct(b, reverse):
        def exponent(x, y):
            return b*x*y - 0.5*b*b*(y*y if reverse else x*x)
        known = fsum(exponent(a, c) for a, c in rows[:-1])
        return exp(known) * step / sqrt(2*pi) * fsum(
            exp(-0.5*y*y + exponent(x, y)) for y in nodes
        )
    weights = (1., *(exp(log_uniform_integral(lambda b: log(direct(b, reverse)), 512)) for reverse in (False, True)))
    expected = [w / fsum(weights) for w in weights]
    assert all(isclose(a, b, abs_tol=1e-12, rel_tol=0) for a, b in zip(observed, expected))
    assert all(abs(a-b) < 1e-7 for a,b in zip(observed, graph_probabilities(rows, panels=1024)))
    print('empty, X -> Y, Y -> X:', *observed)
