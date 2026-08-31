# Causal Inference Toolkit

A teaching **seminar deck** on causal inference with observational and
quasi-experimental data by **Minha Hwang (Dr. Square)**, paired with a
**runnable Python companion** that reproduces every method in the deck on
open-source public data.

The deck walks through the modern causal-inference toolkit — from the
identification-vs-estimation distinction to matching, instrumental variables,
difference-in-differences, regression discontinuity, double/debiased machine
learning, heterogeneous treatment effects, and synthetic control — always with
an emphasis on **which assumptions each method requires** and what it can and
cannot solve.

## Repository layout

```
.
├── 260822.Causal_Infererence_Toolkit-vPublic.md          # Original deck (raw slide export)
├── 260822.Causal_Infererence_Toolkit-vPublic.pdf         # Original deck (PDF)
├── 260822.Causal_Infererence_Toolkit-vPublic_vF.pdf      # Final PDF
├── 260822.Causal_Inference_Toolkit-vPublic_REVISED.md    # Rewritten, clean seminar deck
├── 260822.Causal_Inference_Toolkit-vPublic_change_summary.md  # Technical critique & change log
├── outputs/                                              # Generated artifacts (e.g. .pptx)
└── causal-inference-examples/                            # Runnable Python companion (see its README)
    ├── data/loaders.py            # Cached fetchers for every public dataset
    ├── src/citk_examples/         # Shared helpers (rpy2 bridge, plotting)
    ├── examples/                  # One runnable .py script per method
    ├── notebooks/                 # Narrative notebook that runs all methods
    └── tests/                     # Smoke tests: every example runs end-to-end
```

## The deck

| File | What it is |
|------|-----------|
| `260822.Causal_Infererence_Toolkit-vPublic.md` / `.pdf` | The **original** slide export (raw text; some equations/tables are garbled by the export). |
| `260822.Causal_Infererence_Toolkit-vPublic_vF.pdf` | The **final** presentation PDF. |
| `260822.Causal_Inference_Toolkit-vPublic_REVISED.md` | A **rewritten**, clean Markdown seminar deck with corrected assumptions and current toolkit guidance. |
| `260822.Causal_Inference_Toolkit-vPublic_change_summary.md` | A **technical critique and change log** documenting the factual corrections and structural improvements made in the revised deck. |
| `outputs/` | Generated artifacts such as a `.pptx` rendering of the deck. |

Start with the **revised deck** for the cleanest read, and the **change summary**
to see how it differs from the original.

## The runnable companion — `causal-inference-examples/`

Every method in the deck gets a self-contained, reproducible example on
open-source public data, following the same narrative arc throughout:

> **assumption → estimator → diagnostic → interpretation**

Where a method has no mature Python implementation, the canonical **R** package
is called from Python via [`rpy2`](https://rpy2.github.io/) rather than
re-deriving it.

| # | Method | Package used | Dataset |
|---|--------|--------------|---------|
| 1 | Matching (greedy NN, propensity + caliper) | R `MatchIt` via `rpy2` | LaLonde NSW |
| 2 | Propensity score matching + IPTW | `statsmodels` | LaLonde NSW |
| 3 | Instrumental variables (2SLS) | `linearmodels` | Card college proximity |
| 4 | Difference-in-differences (2×2) | `statsmodels` | Organ donations (CA registry) |
| 5 | Staggered DiD (Callaway & Sant'Anna) | `csdid` | `mpdta` |
| 6 | Regression discontinuity | `rdrobust` | Gov. transfers (Honduras) |
| 7 | Double / debiased ML | `doubleml` | 401(k) participation |
| 8 | DML-IV | `econml` | 401(k) |
| 9 | Heterogeneous effects / CATE (Causal Forest) | `econml` | 401(k) |
| 10 | Synthetic control | `pysyncon` | German reunification |
| 11 | Synthetic control (ML-regularized) | Microsoft `SparseSC` | German reunification |

See [`causal-inference-examples/README.md`](causal-inference-examples/README.md)
for full details and [`causal-inference-examples/data/README.md`](causal-inference-examples/data/README.md)
for dataset provenance and licenses.

## Quick start

```bash
cd causal-inference-examples

# Core install (examples 2, 3, 4, 6, 7, 8, 9, 10)
python -m pip install -e .

# Run a single method
python examples/03_instrumental_variables.py

# Or run every method with explanations in one place
jupyter notebook notebooks/causal_inference_examples.ipynb
```

Some methods rely on optional back-ends:

```bash
python -m pip install -e ".[r]"         # example 1  — R + MatchIt via rpy2
python -m pip install -e ".[sparsesc]"  # example 11 — Microsoft SparseSC (from GitHub)
python -m pip install -e ".[did]"       # example 5  — sibling csdid package
python -m pip install -e ".[dev]"       # pytest + jupyter
```

The notebook and the standalone scripts stay in sync: each notebook cell simply
loads the corresponding `examples/NN_*.py` script and calls its `run()`. If an
optional back-end isn't installed, that section is skipped gracefully so the
notebook still runs top-to-bottom.

## Tests

Smoke tests confirm every example runs end-to-end on its public dataset,
skipping any example whose optional dependency is not installed:

```bash
cd causal-inference-examples
pytest
```

## Author

Deck and materials by **Minha Hwang (Dr. Square)**. Please retain attribution
when reusing the content.
