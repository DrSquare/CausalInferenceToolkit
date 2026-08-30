"""Example 02 — Propensity Score Matching + IPTW.

Deck reference: "Propensity score matching avoids curse of dimensionality",
"Estimating propensity score", "Overlap", and the covariate-balance (SMD) slides.

Confounding from *observed* covariates biases a naive treated-vs-control
comparison. We estimate the propensity score e(X) = P(T=1 | X) with logistic
regression, then recover the ATT two ways the deck describes:

  1. Inverse-probability-of-treatment weighting (IPTW / IPW), and
  2. 1:1 greedy nearest-neighbour matching on the propensity score.

We report standardized mean differences (SMD) before and after matching — the
deck's rule of thumb is |SMD| < 0.1 indicates adequate balance.

Dataset: LaLonde NSW job training. Outcome ``re78`` (1978 earnings), treatment
``treat``. Note: on the *observational* LaLonde sample, adjustment famously does
not fully recover the experimental benchmark — a feature worth discussing.

Run:
    python examples/02_propensity_iptw.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import loaders  # noqa: E402

COVARIATES = ["age", "educ", "married", "nodegree", "re74", "re75", "black", "hispan"]


def _smd(treated: pd.DataFrame, control: pd.DataFrame, cols) -> dict:
    """Standardized mean difference per covariate (pooled-SD denominator)."""
    out = {}
    for c in cols:
        mt, mc = treated[c].mean(), control[c].mean()
        sd = np.sqrt((treated[c].var() + control[c].var()) / 2)
        out[c] = 0.0 if sd == 0 else (mt - mc) / sd
    return out


def run() -> dict:
    df = loaders.load_lalonde().copy()
    # race factor -> indicator columns used as covariates.
    df["black"] = (df["race"] == "black").astype(int)
    df["hispan"] = (df["race"] == "hispan").astype(int)

    y = df["re78"]
    t = df["treat"]

    # --- Naive (unadjusted) difference in means --------------------------------
    naive = y[t == 1].mean() - y[t == 0].mean()

    # --- Propensity score via logistic regression ------------------------------
    X = sm.add_constant(df[COVARIATES])
    pscore = sm.Logit(t, X).fit(disp=0).predict(X)
    df["pscore"] = pscore

    # --- IPTW / IPW estimate of the ATT (Hajek / stabilized) -------------------
    w = np.where(t == 1, 1.0, pscore / (1.0 - pscore))
    treated_mean = np.average(y[t == 1], weights=w[t == 1])
    control_mean = np.average(y[t == 0], weights=w[t == 0])
    iptw_att = treated_mean - control_mean

    # --- 1:1 greedy nearest-neighbour matching on the propensity score ---------
    treated = df[t == 1]
    control = df[t == 0].copy()
    matched_control_idx = []
    available = control.index.tolist()
    for _, row in treated.iterrows():
        dists = (control.loc[available, "pscore"] - row["pscore"]).abs()
        best = dists.idxmin()
        matched_control_idx.append(best)
        available.remove(best)  # matching without replacement
    matched_control = df.loc[matched_control_idx]
    psm_att = treated["re78"].mean() - matched_control["re78"].mean()

    # --- Covariate balance (SMD) before vs after matching ----------------------
    smd_before = _smd(treated, df[t == 0], COVARIATES)
    smd_after = _smd(treated, matched_control, COVARIATES)
    max_smd_before = max(abs(v) for v in smd_before.values())
    max_smd_after = max(abs(v) for v in smd_after.values())

    print("ATT of NSW job training on 1978 earnings (LaLonde, observational)")
    print(f"  Naive difference : {naive:10.1f}")
    print(f"  IPTW ATT         : {iptw_att:10.1f}")
    print(f"  PSM (1:1) ATT    : {psm_att:10.1f}")
    print(f"  Max |SMD| before matching: {max_smd_before:.3f}")
    print(f"  Max |SMD| after  matching: {max_smd_after:.3f} "
          f"({'balanced' if max_smd_after < 0.1 else 'still imbalanced'})")

    return {
        "naive": float(naive),
        "iptw_att": float(iptw_att),
        "psm_att": float(psm_att),
        "max_smd_before": float(max_smd_before),
        "max_smd_after": float(max_smd_after),
    }


if __name__ == "__main__":
    res = run()
    assert np.isfinite(res["iptw_att"]) and np.isfinite(res["psm_att"])
