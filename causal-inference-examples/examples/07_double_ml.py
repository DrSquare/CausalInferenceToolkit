"""Example 07 — Double / Debiased Machine Learning (ATE).

Deck reference: "Double/Debiased/Orthogonal Machine Learning (1/4-4/4)".

DML uses flexible ML to partial out high-dimensional confounders from both the
outcome and the treatment (the "residual-on-residual" idea / FWL on steroids),
then estimates the causal effect from the residuals. Neyman-orthogonality plus
cross-fitting make the estimate robust to the ML models' regularization and
over-fitting biases.

We estimate the ATE of 401(k) *eligibility* (``e401``) on net financial assets
(``net_tfa``) with a partially-linear DML model (random-forest nuisances,
cross-fitted), using the ``doubleml`` package.

Dataset: 401(k) data (Chernozhukov & Hansen) via ``doubleml``.

Run:
    python examples/07_double_ml.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import loaders  # noqa: E402

COVARIATES = ["age", "inc", "fsize", "educ", "db", "marr", "twoearn", "pira", "hown"]


def run() -> dict:
    df = loaders.load_401k().copy()

    dml_data = DoubleMLData(
        df, y_col="net_tfa", d_cols="e401", x_cols=COVARIATES
    )

    # Nuisance learners: outcome regression g(X) and propensity m(X).
    ml_g = RandomForestRegressor(n_estimators=100, max_depth=6,
                                 min_samples_leaf=20, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=100, max_depth=6,
                                  min_samples_leaf=20, random_state=0)

    dml_plr = DoubleMLPLR(dml_data, ml_g, ml_m, n_folds=5)
    dml_plr.fit()

    coef = float(dml_plr.coef[0])
    se = float(dml_plr.se[0])
    ci = dml_plr.confint().iloc[0]

    print("Double ML: ATE of 401(k) eligibility on net financial assets")
    print(f"  ATE estimate : {coef:,.1f}")
    print(f"  Std. error   : {se:,.1f}")
    print(f"  95% CI       : [{ci.iloc[0]:,.1f}, {ci.iloc[1]:,.1f}]")

    return {"ate": coef, "se": se}


if __name__ == "__main__":
    r = run()
    assert np.isfinite(r["ate"])
