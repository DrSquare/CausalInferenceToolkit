"""Shared plotting helpers used across the example scripts.

All functions save to a file path and return the Matplotlib ``Axes`` so the
examples stay headless-friendly (they never call ``plt.show()``).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: examples save figures, never open a window
import matplotlib.pyplot as plt


def event_study_plot(event_times, atts, lower, upper, path, title="Event study"):
    """Plot dynamic (event-time) treatment effects with a confidence band.

    Used by the staggered-DiD example to visualise the parallel-trends pre-test
    (effects left of period 0 should hover around zero) and the post-treatment
    dynamic ATT path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="grey", ls="--", lw=0.8)
    ax.errorbar(
        event_times, atts,
        yerr=[[a - lo for a, lo in zip(atts, lower)],
              [hi - a for a, hi in zip(atts, upper)]],
        fmt="none", ecolor="grey", capsize=3, lw=1.5,
    )
    # Colour points by pre- (red) vs post-treatment (blue) event time.
    colors = ["tab:red" if e < 0 else "tab:blue" for e in event_times]
    ax.scatter(event_times, atts, c=colors, zorder=3)
    ax.set_xlabel("Event time (periods since treatment)")
    ax.set_ylabel("ATT")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return ax
