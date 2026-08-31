# Causal Inference Toolkit — Python Examples

A companion, runnable Python repository for the *Causal Inference Tool Kits (with
Observational / Quasi-Experimental Data)* deck by Minha Hwang. Each method in the
deck gets a self-contained, reproducible example on **open-source public data**.

Where a method has no mature Python implementation, we call the canonical **R**
package from Python via [`rpy2`](https://rpy2.github.io/) instead of re-deriving it.

## Method index

| # | Method | Package used | Dataset | Status |
|---|--------|--------------|---------|--------|
| 1 | Matching (greedy NN, propensity + caliper) | R `MatchIt` via `rpy2` | LaLonde NSW | **working** |
| 2 | Propensity score matching + IPTW | `statsmodels` | LaLonde NSW | **working** |
| 3 | Instrumental variables (2SLS) | `linearmodels` | Card college proximity | **working** |
| 4 | Difference-in-differences (2×2) | `statsmodels` | Organ donations (CA registry) | **working** |
| 5 | Staggered DiD (Callaway & Sant'Anna) | `csdid` (sibling package) | `mpdta` | **working** |
| 6 | Regression discontinuity | `rdrobust` | Gov. transfers (Honduras) | **working** |
| 7 | Double / debiased ML | `doubleml` | 401(k) participation | **working** |
| 8 | DML-IV | `econml` | 401(k) | **working** |
| 9 | Heterogeneous effects / CATE (Causal Forest) | `econml` | 401(k) | **working** |
| 10 | Synthetic control | `pysyncon` | German reunification | **working** |
| 11 | Synthetic control (ML-regularized) | Microsoft `SparseSC` | German reunification | **working** |

Each example follows the same narrative arc used throughout the deck:
**assumption → estimator → diagnostic plot → interpretation.**

## Quick start

```bash
python -m pip install -e .
python examples/03_instrumental_variables.py
python examples/05_did_staggered.py
```

The R-backed examples (1, and optional parity checks in 5) additionally require R
plus `rpy2`. Install the optional extra and a working R:

```bash
python -m pip install -e ".[r]"
```

Example 11 uses Microsoft's `SparseSC`, which is published on GitHub rather than
PyPI (a small compatibility shim, `src/citk_examples/_sparsesc_compat.py`, lets
it run on current SciPy/scikit-learn):

```bash
python -m pip install -e ".[sparsesc]"
```

See [`data/README.md`](data/README.md) for dataset provenance and licenses.

## Layout

```
data/loaders.py          Cached fetchers for every public dataset
src/citk_examples/       Shared helpers (rpy2 bridge, plotting)
examples/                Runnable .py scripts, one per method
notebooks/               Notebook figures + the notebook build helper
tests/                   Smoke tests: every example runs end-to-end
```

The narrative notebook that runs every method lives at the repository root:
[`../causal_inference_examples.ipynb`](../causal_inference_examples.ipynb)
(regenerate it with `python notebooks/_build_notebook.py`).
