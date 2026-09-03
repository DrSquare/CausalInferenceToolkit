# Causal Inference Toolkit

A teaching **seminar deck** on causal inference with observational and
quasi-experimental data by **Minha Hwang (Dr. Square)**, paired with a
**runnable Python companion** that reproduces every method in the deck on
open-source public data.

The deck walks through the modern causal-inference toolkit — from the
identification-vs-estimation distinction to matching, instrumental variables,
difference-in-differences, regression discontinuity, double/debiased machine
learning, heterogeneous treatment effects (Causal Forest **and** S-/T-/X-Learner
meta-learners), and synthetic control — always with an emphasis on **which
assumptions each method requires** and what it can and cannot solve.

## Repository layout

```
.
├── causal_inference_examples.ipynb              # Narrative notebook: runs all 12 methods (start here)
├── 260822.Causal_Infererence_Toolkit-vShare.pdf # The shareable seminar deck (PDF)
├── outputs/                                     # Generated artifacts (e.g. a .pptx rendering of the deck)
└── causal-inference-examples/                   # Runnable Python companion (see its README)
    ├── data/loaders.py            # Cached fetchers for every public dataset
    ├── src/citk_examples/         # Shared helpers (rpy2 bridge, plotting)
    ├── examples/                  # One runnable .py script per method (01–12)
    ├── notebooks/                 # Notebook figures + the notebook build helper
    └── tests/                     # Smoke tests: every example runs end-to-end
```

## The deck

The shareable seminar deck is
[`260822.Causal_Infererence_Toolkit-vShare.pdf`](260822.Causal_Infererence_Toolkit-vShare.pdf)
at the repository root, and a PowerPoint rendering lives in
[`outputs/`](outputs/). The runnable companion below reproduces every method the
deck covers on open-source public data.

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
| 12 | Meta-learners for HTE (S-, T-, X-Learner) | `econml` | 401(k) |

### Heterogeneous treatment effects (HTE): Causal Forest vs. meta-learners

Examples **9** and **12** both estimate the **conditional average treatment
effect** $CATE(x) = E[Y(1) - Y(0)\mid X = x]$ — *who* benefits most, not just the
average. They take two different routes on the same 401(k) data:

- **Example 9 — Causal Forest** (`econml.dml.CausalForestDML`): a Generalized
  Random Forest that splits to maximize the *difference in treatment effect*,
  with valid confidence intervals for the CATE.
- **Example 12 — Meta-learners** (`econml.metalearners`): a recipe that wires
  ordinary supervised regressors together (Künzel et al. 2019).

| Meta-learner | Idea | Pros | Cons |
|--------------|------|------|------|
| **S-Learner** ("Single") | One model $\mu(X, T)$ with treatment as a feature; $CATE=\mu(x,1)-\mu(x,0)$. | Simplest; can shrink CATE to exactly 0 when there's no effect. | The lone model can *wash out* a weak treatment signal, biasing CATE toward 0. |
| **T-Learner** ("Two") | Separate models per arm, $\mu_1(X)$ and $\mu_0(X)$; $CATE=\mu_1(x)-\mu_0(x)$. | Each arm fully flexible; easy to reason about. | No borrowing of strength across arms; high variance with small/imbalanced treated groups. |
| **X-Learner** ("Cross") | Two-stage refinement of the T-Learner, combined with propensity weights. | Efficient under imbalance; usually most robust. | Most moving parts (outcome + effect + propensity models) to tune. |

**Causal Forest vs. meta-learners.** The forest is a single, self-tuning
estimator with built-in inference, and is a strong default when you mainly want
$CATE(x)$ and its uncertainty. Meta-learners are more transparent and let you
plug in *any* regressor, but they shift the modeling choices (and the bias/variance
trade-offs above) onto you. On the 401(k) data the S-Learner reports the smallest
ATE (regularization pulls the effect toward 0), while the T- and X-Learners agree
more closely with each other and with the Causal Forest.

See [`causal-inference-examples/README.md`](causal-inference-examples/README.md)
for full details and [`causal-inference-examples/data/README.md`](causal-inference-examples/data/README.md)
for dataset provenance and licenses.

## Quick start

```bash
# Install the runnable companion (examples 2, 3, 4, 6, 7, 8, 9, 12)
python -m pip install -e causal-inference-examples

# Run a single method
python causal-inference-examples/examples/03_instrumental_variables.py

# Or run every method (all 12) with explanations in one place (from the repo root)
jupyter notebook causal_inference_examples.ipynb
```

Some methods rely on optional back-ends:

```bash
python -m pip install -e "causal-inference-examples[r]"         # example 1       — R + MatchIt via rpy2
python -m pip install -e "causal-inference-examples[ml]"        # examples 8,9,12 — econml (DML-IV, Causal Forest, meta-learners)
python -m pip install -e "causal-inference-examples[sparsesc]"  # example 11      — Microsoft SparseSC (from GitHub)
python -m pip install -e "causal-inference-examples[did]"       # example 5       — sibling csdid package
python -m pip install -e "causal-inference-examples[dev]"       # pytest + jupyter
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
