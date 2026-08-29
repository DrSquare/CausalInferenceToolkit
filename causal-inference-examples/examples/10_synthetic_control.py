"""Example 10 — Synthetic Control.

Deck reference: "Synthetic Control: Better Version of Matching" (listed under
reduced-form causal methods and other topics).

When a single treated unit is observed over time, synthetic control builds a
weighted combination of untreated "donor" units that best reproduces the treated
unit's *pre-treatment* trajectory. The post-treatment gap between the real unit
and its synthetic counterpart estimates the treatment effect.

Classic application (Abadie et al. 2015): the economic cost of the 1990 German
reunification, using West German per-capita GDP vs. a synthetic West Germany
built from other OECD countries.

Dataset: mirrored from the ``pysyncon`` examples.

Run:
    python examples/10_synthetic_control.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pysyncon import Dataprep, Synth

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import loaders  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "figures"
TREATED = "West Germany"
TREATMENT_YEAR = 1990


def run() -> dict:
    df = loaders.load_germany_reunification().copy()
    controls = sorted(set(df["country"].unique()) - {TREATED})

    dataprep = Dataprep(
        foo=df,
        predictors=["gdp", "trade", "infrate"],
        predictors_op="mean",
        time_predictors_prior=range(1971, 1991),
        special_predictors=[
            ("industry", range(1971, 1991), "mean"),
            ("schooling", [1970, 1975], "mean"),
            ("invest70", [1980], "mean"),
        ],
        dependent="gdp",
        unit_variable="country",
        time_variable="year",
        treatment_identifier=TREATED,
        controls_identifier=controls,
        time_optimize_ssr=range(1981, 1991),
    )

    synth = Synth()
    synth.fit(dataprep)

    # Average post-reunification gap (actual - synthetic West Germany).
    att = synth.att(time_period=range(TREATMENT_YEAR + 1, 2004))
    att_value = float(att["att"])

    # Largest donor weights making up the synthetic control.
    weights = synth.weights().sort_values(ascending=False)
    top = weights[weights > 0.01]

    print("Synthetic control: GDP cost of the 1990 German reunification")
    print(f"  Average post-1990 ATT (GDP per capita): {att_value:,.1f}")
    print("  Top donor weights for 'synthetic West Germany':")
    for country, w in top.items():
        print(f"    {country:<15} {w:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        # pysyncon's path_plot calls plt.show() internally; neutralise it so the
        # headless backend doesn't warn, then save the figure ourselves.
        _orig_show = plt.show
        plt.show = lambda *a, **k: None
        try:
            synth.path_plot(time_period=range(1960, 2004),
                            treatment_time=TREATMENT_YEAR)
            plt.title("West Germany vs. synthetic control")
            plt.savefig(FIG_DIR / "10_synthetic_control.png", dpi=120,
                        bbox_inches="tight")
        finally:
            plt.show = _orig_show
            plt.close("all")
    except Exception as exc:
        print(f"  (skipped path plot: {exc})")

    return {"att": att_value, "n_donors": int((weights > 0.01).sum())}


if __name__ == "__main__":
    r = run()
    assert np.isfinite(r["att"])
