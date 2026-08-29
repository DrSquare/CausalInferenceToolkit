"""Example 01 — Matching (via R's MatchIt through rpy2).

Deck reference: the "Matching" slides — greedy (nearest-neighbour) vs. optimal
matching, distance metrics (propensity score, Mahalanobis), calipers, and
checking covariate balance with standardized mean differences (SMD).

The deck explicitly notes there is no mature Python matching package and
recommends running R's ``MatchIt`` from Python via ``rpy2``. This example does
exactly that: it estimates the ATT of NSW job training on 1978 earnings using

  1. greedy nearest-neighbour matching on the propensity score, and
  2. nearest-neighbour matching on Mahalanobis distance with a caliper,

and reports the max |SMD| before vs. after matching (rule of thumb: < 0.1 is
balanced). All numerical work is done in R; only scalars cross back to Python.

Requires R + the ``MatchIt`` package and ``rpy2`` (``pip install -e ".[r]"``).
``_rutils`` auto-detects a local R install and configures ``R_HOME``/``PATH``.

Run:
    python examples/01_matching.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import loaders  # noqa: E402
from citk_examples import _rutils  # noqa: E402

FORMULA = ("treat ~ age + educ + married + nodegree + re74 + re75 + black + hispan")


def _fit_matchit(ro, name: str, method: str, distance: str, extra: str = "") -> dict:
    """Run one matchit() spec in R and return ATT + balance scalars."""
    ro.r(f'''
        {name} <- matchit({FORMULA}, data = d,
                          method = "{method}", distance = "{distance}"{extra})
        {name}_s  <- summary({name})
        {name}_md <- match.data({name})
        {name}_fit <- lm(re78 ~ treat, data = {name}_md, weights = weights)
    ''')
    att = float(ro.r(f'coef({name}_fit)["treat"]')[0])
    smd_before = float(ro.r(f'max(abs({name}_s$sum.all[,"Std. Mean Diff."]))')[0])
    smd_after = float(ro.r(f'max(abs({name}_s$sum.matched[,"Std. Mean Diff."]))')[0])
    return {"att": att, "smd_before": smd_before, "smd_after": smd_after}


def run() -> dict:
    _rutils.require_r()
    _rutils.ensure_packages(["MatchIt"])

    import rpy2.robjects as ro

    df = loaders.load_lalonde().copy()
    df["black"] = (df["race"] == "black").astype(int)
    df["hispan"] = (df["race"] == "hispan").astype(int)
    keep = ["treat", "age", "educ", "married", "nodegree", "re74", "re75",
            "re78", "black", "hispan"]

    ro.globalenv["d"] = _rutils.to_r(df[keep])
    ro.r("suppressMessages(library(MatchIt))")

    # 1) Greedy nearest-neighbour on the propensity score (logistic distance).
    nn = _fit_matchit(ro, "m_nn", method="nearest", distance="glm")
    # 2) Greedy NN on the propensity score with a 0.2-SD caliper (tighter match,
    #    at the cost of dropping poorly-matched treated units).
    cal = _fit_matchit(ro, "m_cal", method="nearest", distance="glm",
                       extra=", caliper = 0.2")

    print("ATT of NSW job training on 1978 earnings (LaLonde) via R MatchIt")
    print("  Greedy NN on propensity score:")
    print(f"    ATT = {nn['att']:8.1f}   max|SMD| {nn['smd_before']:.3f} -> "
          f"{nn['smd_after']:.3f}")
    print("  Greedy NN on propensity score + 0.2-SD caliper:")
    print(f"    ATT = {cal['att']:8.1f}   max|SMD| {cal['smd_before']:.3f} -> "
          f"{cal['smd_after']:.3f}")

    return {"nearest": nn, "caliper": cal}


if __name__ == "__main__":
    res = run()
    assert res["nearest"]["smd_after"] <= res["nearest"]["smd_before"]
