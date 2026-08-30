"""Example 11 — Synthetic control with Microsoft's SparseSC.

Deck reference: the "Synthetic control" slides. Example 10 builds the classic
Abadie synthetic control with ``pysyncon``; this example showcases Microsoft's
open-source **SparseSC** (https://github.com/microsoft/SparseSC), an
ML-enhanced synthetic control that adds L1/L2 regularization on both the unit
weights (W) and the feature weights (V). Regularization is tuned by
cross-validation, which guards against overfitting, yields a unique solution,
and scales to many units/features.

We reuse the canonical German reunification panel (per-capita GDP for 16 OECD
donor countries + West Germany, 1960-2003; treatment = reunification in 1990)
so the SparseSC estimate can be read side-by-side with the classic SC in
example 10. ``estimate_effects`` also runs placebo (permutation) inference,
giving p-values for the average post-treatment gap.

SparseSC 0.2.0 predates current SciPy/scikit-learn; ``_sparsesc_compat.patch()``
transparently bridges the removed/renamed APIs (see that module for details).

Requires ``SparseSC`` (``pip install "git+https://github.com/microsoft/SparseSC.git"``).

Run:
    python examples/11_synthetic_control_sparsesc.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import loaders  # noqa: E402
from citk_examples import _sparsesc_compat  # noqa: E402

TREAT_UNIT = "West Germany"
TREAT_YEAR = 1990


def run() -> dict:
    _sparsesc_compat.patch()
    import SparseSC

    df = loaders.load_germany_reunification()
    # One row per unit, one column per year; outcome = per-capita GDP.
    wide = df.pivot(index="country", columns="year", values="gdp").sort_index()
    years = list(wide.columns)
    outcomes = wide.values.astype(float)

    t0 = years.index(TREAT_YEAR)  # first treated period (column index)
    treated_idx = list(wide.index).index(TREAT_UNIT)
    # NaN for every never-treated donor; the treatment period for the treated.
    unit_treatment_periods = np.full(outcomes.shape[0], np.nan)
    unit_treatment_periods[treated_idx] = t0

    # SparseSC's coordinate-descent Lassos emit many benign non-convergence
    # notes on raw GDP levels; they don't affect the fitted weights here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = SparseSC.estimate_effects(
            outcomes, unit_treatment_periods, fast=True
        )

    post = est.pl_res_post.avg_joint_effect
    pre = est.pl_res_pre.avg_joint_effect
    att = float(post.effect)
    p_value = float(post.p)
    # Per-period treated-vs-synthetic gap over the post-treatment years.
    path = [float(x) for x in np.asarray(est.pl_res_post.effect_vec.effect).ravel()]

    print("Synthetic control via Microsoft SparseSC (ML-regularized)")
    print(f"  Outcome: per-capita GDP; treated: {TREAT_UNIT} @ {TREAT_YEAR}")
    print(f"  Pre-period placebo fit gap: {float(pre.effect):8.1f} "
          f"(p={float(pre.p):.3f})  <- want non-significant (good match)")
    print(f"  Avg post-period effect (ATT): {att:8.1f} (placebo p={p_value:.3f})")
    print(f"  Effect path: first post-year {path[0]:.0f} -> "
          f"last {path[-1]:.0f}")
    print("  Compare example 10 (classic pysyncon SC): ~ -1506")

    return {"att": att, "p_value": p_value, "path": path,
            "pre_fit_gap": float(pre.effect)}


if __name__ == "__main__":
    res = run()
    # Reunification is estimated to have lowered West German GDP vs. its
    # regularized synthetic control.
    assert res["att"] < 0
