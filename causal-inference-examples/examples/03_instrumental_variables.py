"""Example 03 — Instrumental Variables (2SLS).

Deck reference: "Instrumental variables: Estimation - 2SLS" and the Card college
proximity example.

Question: what is the causal return to an extra year of schooling on wages?
OLS is biased because schooling is confounded by unobserved ability. Card (1995)
uses *growing up near a 4-year college* (``nearc4``) as an instrument: it nudges
people toward more schooling (relevance) but is argued not to affect wages except
through schooling (exclusion restriction).

We contrast a naive OLS estimate with the 2SLS estimate and run the standard
first-stage / weak-instrument diagnostic.

Run:
    python examples/03_instrumental_variables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import loaders  # noqa: E402


# Controls used in Card (1995): experience, race, region and metro dummies.
CONTROLS = ["exper", "expersq", "black", "south", "smsa", "reg661", "reg662",
            "reg663", "reg664", "reg665", "reg666", "reg667", "reg668"]


def run() -> dict:
    df = loaders.load_card().copy()
    df = df.dropna(subset=["lwage", "educ", "nearc4"] + CONTROLS)

    y = df["lwage"]
    controls = sm.add_constant(df[CONTROLS])

    # --- Naive OLS (biased by unobserved ability) -----------------------------
    ols = IV2SLS(y, controls.assign(educ=df["educ"]), None, None).fit()
    ols_beta = float(ols.params["educ"])

    # --- 2SLS: instrument educ with nearc4 ------------------------------------
    iv = IV2SLS(
        dependent=y,
        exog=controls,
        endog=df[["educ"]],
        instruments=df[["nearc4"]],
    ).fit()
    iv_beta = float(iv.params["educ"])

    # --- First-stage weak-instrument diagnostic -------------------------------
    first_stage = sm.OLS(
        df["educ"], sm.add_constant(df[CONTROLS + ["nearc4"]])
    ).fit()
    # F-stat for the excluded instrument (rule of thumb: > 10 is "strong").
    t_nearc4 = first_stage.tvalues["nearc4"]
    first_stage_f = float(t_nearc4 ** 2)

    print("Return to schooling (log wage per year of education)")
    print(f"  OLS   : {ols_beta:6.4f}")
    print(f"  2SLS  : {iv_beta:6.4f}")
    print(f"  first-stage F for nearc4: {first_stage_f:6.2f} "
          f"({'strong' if first_stage_f > 10 else 'weak'} instrument)")

    return {
        "ols_beta": ols_beta,
        "iv_beta": iv_beta,
        "first_stage_f": first_stage_f,
        "n": int(len(df)),
    }


if __name__ == "__main__":
    res = run()
    assert np.isfinite(res["iv_beta"])
