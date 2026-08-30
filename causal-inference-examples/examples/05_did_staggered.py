"""Example 05 — Staggered Difference-in-Differences (Callaway & Sant'Anna 2021).

Deck reference: "Advanced Difference in differences (DID): With Staggered
Adoption and Multiple Time Periods."

When units adopt treatment in different periods, the classic two-way fixed-effect
DiD is biased. Callaway & Sant'Anna estimate group-time average treatment effects
ATT(g, t) and aggregate them into clean summaries. This example reuses the
``csdid`` package that lives as a sibling project in this monorepo (the Python
port of the R ``did`` package) on its bundled ``mpdta`` county-employment panel.

We estimate:
  * the overall/simple ATT,
  * the dynamic (event-study) ATT path, which doubles as the parallel-trends
    pre-test (pre-period effects should be near zero).

Run (needs the csdid sibling package):
    python -m pip install -e "../../csdid"
    python examples/05_did_staggered.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# csdid is an in-tree package at the monorepo root (repos/csdid); expose it.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import loaders  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "figures"


def run() -> dict:
    try:
        from csdid.att_gt import ATTgt
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "This example needs the sibling csdid package. Install it with:\n"
            '    python -m pip install -e "../../csdid"'
        ) from exc

    df = loaders.load_mpdta().copy()
    # csdid follows R `did` conventions: gname holds the first treated period,
    # 0 = never treated. Rename the dotted R column for a valid identifier.
    df = df.rename(columns={"first.treat": "first_treat"})

    est = ATTgt(
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first_treat",
        xformla="~lpop",  # covariate; parallel trends holds conditional on lpop
        data=df,
    ).fit(est_method="dr", bstrap=False)

    # --- Simple aggregation: one overall ATT ----------------------------------
    est.aggte(typec="simple")
    simple_att = float(np.asarray(est.atte["overall_att"], dtype=float).ravel()[0])

    # --- Dynamic aggregation: event-study path --------------------------------
    est.aggte(typec="dynamic")
    event_times = np.asarray(est.atte["egt"], dtype=float).ravel()
    dyn_att = np.asarray(est.atte["att_egt"], dtype=float).ravel()

    print("Staggered DiD on mpdta (effect of minimum-wage change on log teen employment)")
    print(f"  Simple overall ATT : {simple_att:7.4f}")
    print("  Dynamic ATT by event time:")
    for e, a in zip(event_times, dyn_att):
        tag = "pre " if e < 0 else "post"
        print(f"    e={e:+.0f} ({tag}): {a:7.4f}")

    try:
        from citk_examples.viz import event_study_plot

        se = np.asarray(est.atte["se_egt"], dtype=float).ravel()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        event_study_plot(
            event_times.tolist(),
            dyn_att.tolist(),
            (dyn_att - 1.96 * se).tolist(),
            (dyn_att + 1.96 * se).tolist(),
            FIG_DIR / "05_did_event_study.png",
            title="Staggered DiD event study (mpdta)",
        )
    except Exception as exc:  # plotting is a nicety, not required for the result
        print(f"  (skipped event-study plot: {exc})")

    return {
        "simple_att": simple_att,
        "event_times": event_times.tolist(),
        "dynamic_att": dyn_att.tolist(),
    }


if __name__ == "__main__":
    res = run()
    assert np.isfinite(res["simple_att"])
