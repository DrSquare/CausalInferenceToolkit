"""Example 09 — Heterogeneous Treatment Effects (Causal Forest / CATE).

Deck reference: "Heterogeneous Treatment Effects (HTE): CATE Estimation (1/4-
4/4) — The Causal Forest Solution".

The ATE hides variation: the same treatment can help some subgroups far more
than others. A Causal Forest (a Generalized Random Forest) estimates the
Conditional Average Treatment Effect CATE(x) by splitting to *maximize
difference in treatment effect* rather than to predict the outcome, using
DML-style residualization and "honest" (cross-fit) leaves.

We estimate how the effect of 401(k) eligibility on net financial assets varies
with income and age using econml's ``CausalForestDML``, then summarize the CATE
across income quartiles.

Dataset: 401(k) data (Chernozhukov & Hansen) via ``doubleml``.

Run:
    python examples/09_heterogeneous_effects.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import loaders  # noqa: E402

# Effect modifiers (X) vs. remaining controls (W).
HET_FEATURES = ["inc", "age"]
CONTROLS = ["fsize", "educ", "db", "marr", "twoearn", "pira", "hown"]


def run() -> dict:
    df = loaders.load_401k().copy()

    Y = df["net_tfa"].values
    T = df["e401"].values
    X = df[HET_FEATURES].values
    W = df[CONTROLS].values

    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=6,
                                      min_samples_leaf=20, random_state=0),
        model_t=RandomForestClassifier(n_estimators=100, max_depth=6,
                                       min_samples_leaf=20, random_state=0),
        discrete_treatment=True,
        n_estimators=500,
        min_samples_leaf=20,
        random_state=0,
    )
    est.fit(Y, T, X=X, W=W)

    ate = float(np.asarray(est.ate(X)).ravel()[0])
    cate = np.asarray(est.effect(X)).ravel()

    # Summarize CATE across income quartiles.
    q = pd.qcut(df["inc"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    by_income = pd.Series(cate).groupby(q.values, observed=False).mean()

    print("Causal Forest: CATE of 401(k) eligibility on net financial assets")
    print(f"  Overall ATE : {ate:,.1f}")
    print("  Mean CATE by income quartile:")
    for label, value in by_income.items():
        print(f"    {label}: {value:,.1f}")

    return {
        "ate": ate,
        "cate_by_income": {str(k): float(v) for k, v in by_income.items()},
    }


if __name__ == "__main__":
    r = run()
    assert np.isfinite(r["ate"])
    assert len(r["cate_by_income"]) == 4
