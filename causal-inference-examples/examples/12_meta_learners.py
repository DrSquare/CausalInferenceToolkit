"""Example 12 — Meta-Learners for HTE: S-, T-, and X-Learner.

Deck reference: "Heterogeneous Treatment Effects (HTE): CATE Estimation" — the
*meta-learner* family that sits alongside the Causal Forest (example 09).

Meta-learners are a recipe, not a model: they estimate the Conditional Average
Treatment Effect CATE(x) = E[Y(1) - Y(0) | X = x] by combining off-the-shelf
supervised regressors ("base learners"). The three canonical strategies
(Kunzel et al. 2019) differ in how they wire those regressors together:

  * **S-Learner ("Single").** Fit ONE model mu(X, T) on the pooled data with the
    treatment T as just another feature, then CATE(x) = mu(x, 1) - mu(x, 0).
  * **T-Learner ("Two").** Fit TWO separate models, mu_1(X) on the treated and
    mu_0(X) on the controls, then CATE(x) = mu_1(x) - mu_0(x).
  * **X-Learner ("Cross").** A two-stage refinement of the T-Learner: impute
    individual treatment effects by crossing each arm's model onto the other
    arm, fit a second-stage model to those imputed effects in each arm, then
    combine the two using the propensity score as the weight. Designed to shine
    when the treated and control groups are very different in size.

Pros and cons (also summarised in the notebook and README):

  S-Learner  + simplest; one model; naturally handles no-effect (can shrink CATE
               all the way to 0).
             - the single model can "wash out" a weak treatment signal, biasing
               CATE toward 0 when T is one feature among many strong controls.
  T-Learner  + each arm modelled with full flexibility; no shared-form
               restriction; easy to reason about.
             - no borrowing of strength across arms; high variance and unstable
               with imbalanced arms or small treated groups; the two regressors'
               errors don't cancel.
  X-Learner  + best of both: efficient under imbalance, borrows strength across
               arms, propensity-weighted; usually the most robust CATE.
             - most moving parts (outcome + effect + propensity models); more to
               tune and to get wrong.

We estimate the CATE of 401(k) *eligibility* (``e401``) on net financial assets
(``net_tfa``) with all three learners, compare their overall ATE (= mean CATE),
and summarise how the CATE varies across income quartiles — so the three can be
read side-by-side, and against the Causal Forest in example 09.

Dataset: 401(k) data (Chernozhukov & Hansen) via ``doubleml``.

Run:
    python examples/12_meta_learners.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import loaders  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "figures"

# All covariates enter as features X (meta-learners have no separate controls W).
COVARIATES = ["inc", "age", "fsize", "educ", "db", "marr", "twoearn", "pira", "hown"]


def _base_regressor() -> RandomForestRegressor:
    """A shared outcome-model spec so the three learners are comparable."""
    return RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=20, random_state=0
    )


def _cate_by_income(df: pd.DataFrame, cate: np.ndarray) -> dict:
    """Mean CATE within income quartiles."""
    q = pd.qcut(df["inc"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    by = pd.Series(cate).groupby(q.values, observed=False).mean()
    return {str(k): float(v) for k, v in by.items()}


def run() -> dict:
    df = loaders.load_401k().copy()

    Y = df["net_tfa"].values
    T = df["e401"].values  # binary treatment: eligibility
    X = df[COVARIATES].values

    # --- S-Learner: one pooled model with T as a feature ----------------------
    s_learner = SLearner(overall_model=_base_regressor())
    s_learner.fit(Y, T, X=X)
    s_cate = np.asarray(s_learner.effect(X)).ravel()

    # --- T-Learner: one model per treatment arm -------------------------------
    t_learner = TLearner(models=_base_regressor())
    t_learner.fit(Y, T, X=X)
    t_cate = np.asarray(t_learner.effect(X)).ravel()

    # --- X-Learner: cross-fitted, propensity-weighted refinement --------------
    x_learner = XLearner(
        models=_base_regressor(),
        cate_models=_base_regressor(),
        propensity_model=make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
    )
    x_learner.fit(Y, T, X=X)
    x_cate = np.asarray(x_learner.effect(X)).ravel()

    results = {
        "s_learner": {"ate": float(s_cate.mean()),
                      "cate_by_income": _cate_by_income(df, s_cate)},
        "t_learner": {"ate": float(t_cate.mean()),
                      "cate_by_income": _cate_by_income(df, t_cate)},
        "x_learner": {"ate": float(x_cate.mean()),
                      "cate_by_income": _cate_by_income(df, x_cate)},
    }

    print("Meta-learners (HTE / CATE) of 401(k) eligibility on net financial assets")
    print(f"{'Learner':<10}{'Overall ATE':>14}   CATE by income quartile (Q1..Q4)")
    for name, key in [("S-Learner", "s_learner"),
                      ("T-Learner", "t_learner"),
                      ("X-Learner", "x_learner")]:
        r = results[key]
        q = r["cate_by_income"]
        qs = "  ".join(f"{q[f'Q{i}']:>9,.0f}" for i in range(1, 5))
        print(f"{name:<10}{r['ate']:>14,.1f}   {qs}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        quartiles = ["Q1", "Q2", "Q3", "Q4"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, key in [("S-Learner", "s_learner"),
                          ("T-Learner", "t_learner"),
                          ("X-Learner", "x_learner")]:
            q = results[key]["cate_by_income"]
            ax.plot(quartiles, [q[k] for k in quartiles], marker="o", label=name)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Income quartile")
        ax.set_ylabel("Mean CATE (effect on net financial assets)")
        ax.set_title("S- vs T- vs X-Learner: CATE of 401(k) eligibility by income")
        ax.legend()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "12_meta_learners.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # plotting is a nicety, not required for the result
        print(f"  (skipped CATE plot: {exc})")

    return results


if __name__ == "__main__":
    res = run()
    for key in ("s_learner", "t_learner", "x_learner"):
        assert np.isfinite(res[key]["ate"])
        assert len(res[key]["cate_by_income"]) == 4
