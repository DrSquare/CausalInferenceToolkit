"""Example 04 — Difference-in-Differences (canonical 2x2).

Deck reference: "Difference in differences (DID): Quasi-Experiment" and
"Difference in differences (DID): Implemented In Regression".

The 2x2 DiD estimates the treatment effect as the interaction of a
treated-group dummy and a post-period dummy in an OLS regression:

    Y = b0 + b1*Treated + b2*Post + b3*(Treated x Post) + e

b3 is the DiD estimate. Identification rests on the *parallel trends*
assumption: absent treatment, treated and control groups would have moved in
parallel. We also plot the group means over time as an eyeball check.

Dataset: California organ-donor registration (``causaldata``). California
adopted an active-choice online registry in Q3 2011; all other states are
controls. Outcome ``Rate`` = donor registration rate.

Run:
    python examples/04_did_basic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import loaders  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "figures"
# Q42010=1, Q12011=2, Q22011=3, Q32011=4, ... — policy is live from Q3 2011.
TREATMENT_QUARTER_NUM = 4


def run() -> dict:
    df = loaders.load_organ_donations().copy()
    df["treated"] = (df["State"] == "California").astype(int)
    df["post"] = (df["Quarter_Num"] >= TREATMENT_QUARTER_NUM).astype(int)

    # --- 2x2 DiD via the interaction term -------------------------------------
    model = smf.ols("Rate ~ treated * post", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["State"]}
    )
    did = float(model.params["treated:post"])
    did_p = float(model.pvalues["treated:post"])

    # Manual 2x2 cross-check (four cell means).
    def cell(tr, po):
        return df[(df["treated"] == tr) & (df["post"] == po)]["Rate"].mean()

    manual = (cell(1, 1) - cell(1, 0)) - (cell(0, 1) - cell(0, 0))

    print("DiD: effect of California's active-choice registry on donor sign-up rate")
    print(f"  DiD estimate (interaction) : {did:+.4f}  (p = {did_p:.3f})")
    print(f"  Manual 2x2 cross-check     : {manual:+.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        means = (df.groupby(["Quarter_Num", "treated"])["Rate"].mean().unstack())
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(means.index, means[0], marker="o", label="Control states")
        ax.plot(means.index, means[1], marker="o", label="California (treated)")
        ax.axvline(TREATMENT_QUARTER_NUM - 0.5, color="grey", ls="--",
                   label="Policy change")
        ax.set_xlabel("Quarter (1 = Q4 2010)")
        ax.set_ylabel("Donor registration rate")
        ax.set_title("Parallel-trends check (organ donations)")
        ax.legend()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "04_did_parallel_trends.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"  (skipped trends plot: {exc})")

    return {"did": did, "did_pvalue": did_p, "manual_did": float(manual)}


if __name__ == "__main__":
    res = run()
    assert np.isfinite(res["did"])
    assert abs(res["did"] - res["manual_did"]) < 1e-9
