"""Example 06 — Regression Discontinuity Design (sharp RDD).

Deck reference: "Regression Discontinuity Design (RDD)".

A continuous *running variable* determines treatment via a cutoff. Units just
above vs. just below the threshold are "as-if" randomly assigned, so the jump in
the outcome at the cutoff identifies a local average treatment effect (LATE).

Here households with a centered income below 0 are eligible for a conditional
cash transfer (sharp assignment). We estimate the discontinuity in program
support at the cutoff with ``rdrobust`` (local-polynomial estimator with a
data-driven bandwidth and bias-corrected robust inference).

Dataset: ``causaldata`` government transfers (Manacorda, Miguel & Vigorito).

Run:
    python examples/06_regression_discontinuity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdrobust import rdrobust

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import loaders  # noqa: E402


def run() -> dict:
    df = loaders.load_gov_transfers().copy()
    running = df["Income_Centered"]
    outcome = df["Support"]

    # Sharp RDD at cutoff 0; rdrobust selects bandwidth and does robust inference.
    res = rdrobust(y=outcome, x=running, c=0.0)

    # rdrobust exposes coefficients/SE/p-values as small DataFrames indexed by
    # estimator type ("Conventional", "Bias-Corrected", "Robust").
    late = float(res.coef.loc["Conventional"].iloc[0])
    robust_p = float(res.pv.loc["Robust"].iloc[0])
    bandwidth = float(res.bws.loc["h"].iloc[0])

    print("Sharp RDD: effect of cash-transfer eligibility on program support")
    print(f"  LATE at cutoff (conventional) : {late:+.4f}")
    print(f"  Robust p-value                : {robust_p:.4f}")
    print(f"  Bandwidth (h)                 : {bandwidth:.3f}")

    return {"late": late, "robust_pvalue": robust_p, "bandwidth": bandwidth}


if __name__ == "__main__":
    r = run()
    assert np.isfinite(r["late"])
