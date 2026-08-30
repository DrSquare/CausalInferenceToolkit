"""Example 08 — DML-IV (Double ML with an instrument).

Deck reference: "DML-IV: Double Machine Learning + Instrumental Variable".

When the treatment is endogenous, DML-IV combines the DML "triple scrub"
(residualize outcome, treatment, and instrument on covariates with ML) with a
2SLS-style purification: use the residualized instrument to isolate the clean
variation in the treatment, then map it onto the residualized outcome.

Here 401(k) *participation* (``p401``) is endogenous (people who save more may
self-select into participation). Eligibility (``e401``) is the instrument: it is
assigned quasi-randomly through employers and plausibly affects wealth only
through participation. We estimate the LATE of participation on net financial
assets with econml's orthogonal IV (``OrthoIV``).

Dataset: 401(k) data (Chernozhukov & Hansen) via ``doubleml``.

Run:
    python examples/08_dml_iv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from econml.iv.dml import OrthoIV

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import loaders  # noqa: E402

COVARIATES = ["age", "inc", "fsize", "educ", "db", "marr", "twoearn", "pira", "hown"]


def run() -> dict:
    df = loaders.load_401k().copy()

    Y = df["net_tfa"].values
    T = df["p401"].values  # endogenous treatment: participation
    Z = df["e401"].values  # instrument: eligibility
    W = df[COVARIATES].values

    est = OrthoIV(
        discrete_treatment=True,
        discrete_instrument=True,
        model_y_xw="auto",
        model_t_xw="auto",
        model_z_xw="auto",
    )
    est.fit(Y, T, Z=Z, W=W)

    late = float(np.asarray(est.effect()).ravel()[0])
    lo, hi = est.effect_interval()
    lo = float(np.asarray(lo).ravel()[0])
    hi = float(np.asarray(hi).ravel()[0])

    print("DML-IV: LATE of 401(k) participation on net financial assets")
    print("  (eligibility instruments for participation)")
    print(f"  LATE estimate : {late:,.1f}")
    print(f"  95% CI        : [{lo:,.1f}, {hi:,.1f}]")

    return {"late": late, "ci_low": lo, "ci_high": hi}


if __name__ == "__main__":
    r = run()
    assert np.isfinite(r["late"])
