"""Generate causal_inference_examples.ipynb from structured cell content.

Run:  python causal-inference-examples/notebooks/_build_notebook.py
This script is a build helper; the committed artifact is the .ipynb it writes,
which lives at the repository root (the notebook is the main user-facing entry
point, while the example scripts it drives stay in causal-inference-examples/).
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
# notebooks/ -> causal-inference-examples/ -> repository root.
REPO_ROOT = HERE.parents[1]
OUT = REPO_ROOT / "causal_inference_examples.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# --------------------------------------------------------------------------- #
# Title & overview
# --------------------------------------------------------------------------- #
md(r"""
# Causal Inference Toolkit — Runnable Examples

This notebook runs **every method** in the *Causal Inference Tool Kits (with
Observational / Quasi-Experimental Data)* deck by Minha Hwang, one section per
method, each on **open-source public data**.

Every section follows the same narrative arc used throughout the deck:

> **assumption → estimator → diagnostic → interpretation**

Each code cell simply loads the corresponding `examples/NN_*.py` script and calls
its `run()` function, so the notebook and the standalone scripts always stay in
sync. Results (and any diagnostic plots) are printed / displayed inline.

| # | Method | Package | Dataset |
|---|--------|---------|---------|
| 1 | Matching (greedy NN, caliper) | R `MatchIt` via `rpy2` | LaLonde NSW |
| 2 | Propensity score matching + IPTW | `statsmodels` | LaLonde NSW |
| 3 | Instrumental variables (2SLS) | `linearmodels` | Card college proximity |
| 4 | Difference-in-differences (2×2) | `statsmodels` | Organ donations |
| 5 | Staggered DiD (Callaway & Sant'Anna) | `csdid` | `mpdta` |
| 6 | Regression discontinuity | `rdrobust` | Gov. transfers (Honduras) |
| 7 | Double / debiased ML | `doubleml` | 401(k) |
| 8 | DML-IV | `econml` | 401(k) |
| 9 | Heterogeneous effects / CATE | `econml` | 401(k) |
| 10 | Synthetic control | `pysyncon` | German reunification |
| 11 | Synthetic control (ML-regularized) | Microsoft `SparseSC` | German reunification |

> **Note on optional dependencies.** Examples 1 (R + `MatchIt`), 5 (`csdid`),
> and 11 (`SparseSC`) rely on optional back-ends. If one isn't installed, that
> cell prints a short *skipped* message instead of failing, so the rest of the
> notebook still runs top-to-bottom. Install everything with:
>
> ```bash
> python -m pip install -e ".[r,ml,sparsesc,did,dev]"
> ```
""")

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
md(r"""
## Setup

We locate the `causal-inference-examples/` package (this notebook sits at the
repository root) and wire up the import paths its example scripts expect
(`examples/`, `data/`, `src/`, and the monorepo root for the sibling `csdid`
package), then define a small `run_example()` helper that loads a numbered
script by file path — module names like `01_matching` aren't valid Python
identifiers, so we load them the same way the test-suite does
(`importlib.util.spec_from_file_location`).
""")

code(r"""
import importlib
import importlib.util
import sys
import traceback
from pathlib import Path

import matplotlib
# Force a non-interactive backend so example plots are written to files; we then
# display those PNGs inline in the relevant sections.
matplotlib.use("Agg")

# The notebook lives at the repository root; the runnable examples package is in
# ./causal-inference-examples. Locate it robustly regardless of the launch CWD.
PROJECT_ROOT = Path.cwd()
if (PROJECT_ROOT / "causal-inference-examples" / "examples").exists():
    # Notebook run from the repository root (its home).
    PKG = PROJECT_ROOT / "causal-inference-examples"
elif (PROJECT_ROOT / "examples").exists():
    # Notebook run from inside the examples package directory.
    PKG = PROJECT_ROOT
else:
    # Fallback: search upward for the package.
    PKG = next(
        (p / "causal-inference-examples"
         for p in [PROJECT_ROOT, *PROJECT_ROOT.parents]
         if (p / "causal-inference-examples" / "examples").exists()),
        PROJECT_ROOT / "causal-inference-examples",
    )

EXAMPLES = PKG / "examples"
FIG_DIR = PKG / "notebooks" / "figures"

for p in (
    EXAMPLES,
    PKG / "data",
    PKG / "src",
    PKG.parent.parent,  # monorepo root (for the sibling csdid package)
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def run_example(name: str):
    '''Load examples/<name>.py, call its run(), and return the result dict.

    Optional back-ends (R, csdid, SparseSC, ...) may be missing; we catch
    ImportError so the rest of the notebook keeps running.
    '''
    try:
        spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run()
    except ImportError as exc:
        print(f"[skipped] {name}: optional dependency not available -> {exc}")
        return None
    except RuntimeError as exc:
        # Some examples (e.g. the R-backed matching) raise RuntimeError when an
        # optional back-end such as R/rpy2 isn't available; treat as a skip.
        print(f"[skipped] {name}: back-end not available -> {exc}")
        return None
    except Exception as exc:  # keep the notebook flowing, but show what broke
        print(f"[error] {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


print("Repository root:", PROJECT_ROOT)
print("Examples package:", PKG)
""")

code(r"""
from IPython.display import Image, display


def show_figure(filename: str):
    '''Display a diagnostic PNG written by an example, if it exists.'''
    path = FIG_DIR / filename
    if path.exists():
        display(Image(filename=str(path)))
    else:
        print(f"(no figure at {path})")
""")

# --------------------------------------------------------------------------- #
# Per-method sections: (number, filename, title, markdown, post-run code)
# --------------------------------------------------------------------------- #
sections = [
    (
        "01_matching",
        "1 · Matching (greedy NN + caliper, via R `MatchIt`)",
        r"""
**Assumption.** Conditional on observed covariates $X$, treatment is as-good-as
random (*selection on observables* / unconfoundedness) with overlap.

**Estimator.** The deck notes Python lacks a mature matching package and
recommends R's `MatchIt` through `rpy2`. We estimate the ATT of NSW job training
on 1978 earnings two ways: (1) greedy nearest-neighbour on the propensity score,
and (2) the same with a 0.2-SD **caliper** (tighter matches, dropping poor ones).

**Diagnostic.** Max $|\text{SMD}|$ (standardized mean difference) before vs.
after matching — the rule of thumb is $|\text{SMD}| < 0.1$ is balanced.

**Interpretation.** Matching should *shrink* covariate imbalance; the caliper
trades sample size for a tighter match.

> Requires R + the `MatchIt` package and `rpy2` (`pip install -e ".[r]"`). If
> R isn't available this cell is skipped.
""",
        "",
    ),
    (
        "02_propensity_iptw",
        "2 · Propensity Score Matching + IPTW",
        r"""
**Assumption.** Same unconfoundedness + overlap as matching; the propensity
score $e(X)=P(T{=}1\mid X)$ is a sufficient one-dimensional summary of $X$
(avoiding the curse of dimensionality).

**Estimator.** Logistic propensity score, then the ATT two ways: **IPTW**
(inverse-probability-of-treatment weighting) and 1:1 greedy nearest-neighbour
**PSM**. We contrast both with the naive difference in means.

**Diagnostic.** Max $|\text{SMD}|$ before vs. after matching.

**Interpretation.** On the *observational* LaLonde sample, adjustment famously
does **not** fully recover the experimental benchmark — a useful cautionary tale
about relying on selection-on-observables.
""",
        "",
    ),
    (
        "03_instrumental_variables",
        "3 · Instrumental Variables (2SLS)",
        r"""
**Assumption.** The instrument (growing up **near a 4-year college**, `nearc4`)
is *relevant* (predicts schooling) and satisfies the *exclusion restriction*
(affects wages only through schooling).

**Estimator.** Two-stage least squares (`linearmodels.IV2SLS`) for the causal
return to a year of schooling, contrasted with naive OLS (biased by unobserved
ability).

**Diagnostic.** First-stage F-statistic for the excluded instrument — the rule
of thumb is $F > 10$ for a "strong" instrument.

**Interpretation.** IV typically *raises* the estimated return relative to OLS,
consistent with attenuation from ability/measurement issues.
""",
        "",
    ),
    (
        "04_did_basic",
        "4 · Difference-in-Differences (canonical 2×2)",
        r"""
**Assumption.** **Parallel trends** — absent treatment, treated and control
groups would have moved in parallel.

**Estimator.** OLS with a treated×post interaction,
$Y = \beta_0 + \beta_1\text{Treated} + \beta_2\text{Post} + \beta_3(\text{Treated}\times\text{Post})$;
$\beta_3$ is the DiD. Standard errors are clustered by state, and we cross-check
against the manual four-cell 2×2 computation.

**Diagnostic.** Group-mean trajectories over time (eyeball parallel-trends
check), plotted below.

**Interpretation.** $\beta_3$ is the effect of California's active-choice
registry on the donor sign-up rate.
""",
        'show_figure("04_did_parallel_trends.png")',
    ),
    (
        "05_did_staggered",
        "5 · Staggered DiD (Callaway & Sant'Anna 2021)",
        r"""
**Assumption.** (Conditional) parallel trends when units adopt treatment at
**different times** — the setting where classic two-way fixed-effects DiD is
biased by "bad" comparisons.

**Estimator.** Group-time effects $ATT(g,t)$ via the `csdid` package
(doubly-robust), aggregated into (a) a single overall ATT and (b) a dynamic
**event-study** path.

**Diagnostic.** The event-study path doubles as a parallel-trends pre-test:
pre-period ($e<0$) effects should be near zero. Plotted below.

**Interpretation.** The overall ATT summarizes the effect of the minimum-wage
change on log teen employment.

> Requires the sibling `csdid` package (`pip install -e "../csdid"`). Skipped if
> unavailable.
""",
        'show_figure("05_did_event_study.png")',
    ),
    (
        "06_regression_discontinuity",
        "6 · Regression Discontinuity Design (sharp RDD)",
        r"""
**Assumption.** A continuous **running variable** determines treatment at a
cutoff; units just above vs. just below are "as-if" randomly assigned
(continuity of potential outcomes at the threshold).

**Estimator.** Local-polynomial estimator with a data-driven bandwidth and
bias-corrected robust inference (`rdrobust`). Here households with centered
income below 0 are eligible for a conditional cash transfer.

**Diagnostic.** Robust p-value and the selected bandwidth $h$.

**Interpretation.** The jump at the cutoff is a **local** average treatment
effect (LATE) — valid for units near the threshold.
""",
        "",
    ),
    (
        "07_double_ml",
        "7 · Double / Debiased Machine Learning (ATE)",
        r"""
**Assumption.** Unconfoundedness given a **high-dimensional** covariate set;
flexible ML models the nuisances (outcome regression + propensity).

**Estimator.** Partially-linear DML (`doubleml.DoubleMLPLR`) with random-forest
nuisances and **cross-fitting**. Neyman-orthogonality makes the effect robust to
regularization/over-fitting bias — the "residual-on-residual" idea (FWL on
steroids).

**Diagnostic.** Standard error and 95% confidence interval from the orthogonal
score.

**Interpretation.** The ATE of 401(k) **eligibility** on net financial assets
(canonically a large positive number, ~\$9k).
""",
        "",
    ),
    (
        "08_dml_iv",
        "8 · DML-IV (Double ML with an instrument)",
        r"""
**Assumption.** Treatment is **endogenous**; a valid instrument is available.
Here 401(k) **participation** (`p401`) is endogenous and **eligibility**
(`e401`) is the instrument (quasi-randomly assigned by employers, affecting
wealth only through participation).

**Estimator.** Orthogonal IV (`econml.iv.dml.OrthoIV`): residualize outcome,
treatment, and instrument on covariates with ML, then a 2SLS-style purification.

**Diagnostic.** 95% confidence interval around the LATE.

**Interpretation.** The LATE of participation on net financial assets, purged of
self-selection into participation.
""",
        "",
    ),
    (
        "09_heterogeneous_effects",
        "9 · Heterogeneous Effects / CATE (Causal Forest)",
        r"""
**Assumption.** The ATE hides variation; effects differ by covariates $X$. Same
unconfoundedness as DML, plus the forest's "honest" (cross-fit) splitting.

**Estimator.** `econml.dml.CausalForestDML` — a Generalized Random Forest that
splits to **maximize difference in treatment effect** (not to predict the
outcome), estimating $CATE(x)$.

**Diagnostic.** Mean CATE across **income quartiles** (does the effect rise with
income?).

**Interpretation.** Reveals *who* benefits most from 401(k) eligibility —
actionable heterogeneity the ATE alone would miss.
""",
        "",
    ),
    (
        "10_synthetic_control",
        "10 · Synthetic Control",
        r"""
**Assumption.** A single treated unit observed over time can be reproduced by a
**weighted combination of donor units** that matches its *pre-treatment*
trajectory (and the weights stay valid post-treatment).

**Estimator.** Classic Abadie synthetic control (`pysyncon`): the 1990 German
reunification, West German per-capita GDP vs. a "synthetic West Germany" of other
OECD countries.

**Diagnostic.** Actual-vs-synthetic pre-treatment fit (plotted below) and the
top donor weights.

**Interpretation.** The post-1990 gap estimates the GDP cost of reunification.
""",
        'show_figure("10_synthetic_control.png")',
    ),
    (
        "11_synthetic_control_sparsesc",
        "11 · Synthetic Control, ML-regularized (Microsoft `SparseSC`)",
        r"""
**Assumption.** Same synthetic-control setup as example 10, but with
**regularization** ($L_1/L_2$ on both unit weights $W$ and feature weights $V$)
tuned by cross-validation — guarding against over-fitting and giving a unique,
scalable solution.

**Estimator.** Microsoft's `SparseSC.estimate_effects` on the same German
reunification panel, with **placebo (permutation) inference**.

**Diagnostic.** Pre-period placebo gap should be *non-significant* (good match);
post-period average effect comes with a placebo p-value.

**Interpretation.** Read side-by-side with example 10 (~-1506): the
ML-regularized SC also finds a **negative** reunification effect on GDP.

> Requires `SparseSC` (`pip install -e ".[sparsesc]"`). Skipped if unavailable.
""",
        "",
    ),
]

for name, title, explanation, post in sections:
    md(f"## {title}\n\n{explanation.strip()}")
    run_line = f'result_{name.split("_")[0]} = run_example({name!r})'
    body = run_line + ("\n" + post if post else "")
    code(body)

# --------------------------------------------------------------------------- #
# Wrap up
# --------------------------------------------------------------------------- #
md(r"""
## Summary

You've now run the full toolkit end-to-end:

- **Selection-on-observables:** matching, propensity scores + IPTW, Double ML,
  causal forests (CATE).
- **Quasi-experiments:** instrumental variables, DiD (2×2 and staggered),
  regression discontinuity.
- **Comparative case studies:** synthetic control (classic and ML-regularized).

Each method encodes a different **identifying assumption** — always the first
thing to scrutinize. The matching helpers, plots, and datasets live under
`causal-inference-examples/` (`data/loaders.py`, `src/citk_examples/`, and the
`examples/` scripts); see the
[project README](README.md),
[`causal-inference-examples/README.md`](causal-inference-examples/README.md),
and [`causal-inference-examples/data/README.md`](causal-inference-examples/data/README.md)
for provenance and licenses.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}

nbf.write(nb, OUT)
print(f"Wrote {OUT} ({len(cells)} cells)")
