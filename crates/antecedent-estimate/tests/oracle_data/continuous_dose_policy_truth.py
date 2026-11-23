"""Kernel-smoothed truth of a fixed group-dose policy for a known data-generating process.

Two baseline groups of equal size.  Within group g the dose has the known density f_g on [0, 1]
(f_a(d) = 2 d, f_b(d) = 2 (1 - d)) and E[Y | D = d, g] = offset_g + sin(4 d), with offset_a = 1
and offset_b = 2.  A policy assigns dose t_g to group g; its recorded value is

    V = sum_g pi_g  int K((d - t_g) / h) m_g(d) dd  /  int K((d - t_g) / h) dd

with the Epanechnikov kernel K(u) = 0.75 (1 - u^2) on |u| < 1 and group fractions pi_g = 1/2.
This is the quantity the inverse-density kernel ratio estimates when exchangeability holds,
the density is the true one and the window lies inside the support.  The integrals are done by
mpmath at 40 digits; no product code is used.

    uv run --with mpmath python continuous_dose_policy_truth.py
"""

from mpmath import mp, mpf, quad, sin

mp.dps = 40
H = mpf("0.2")
OFFSET = {"a": mpf(1), "b": mpf(2)}


def smoothed(group, target):
    t = mpf(target)

    def kernel(d):
        u = (d - t) / H
        return mpf("0.75") * (1 - u * u)

    num = quad(lambda d: kernel(d) * (OFFSET[group] + sin(4 * d)), [t - H, t + H])
    den = quad(kernel, [t - H, t + H])
    return num / den


def value(policy):
    return sum(smoothed(g, t) for g, t in policy.items()) / 2


policy = {"a": "0.7", "b": "0.3"}
reference = {"a": "0.5", "b": "0.5"}
v_p, v_r = value(policy), value(reference)
print("policy", mp.nstr(v_p, 17))
print("reference", mp.nstr(v_r, 17))
print("incremental", mp.nstr(v_p - v_r, 17))
