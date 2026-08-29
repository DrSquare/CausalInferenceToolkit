# Datasets

All datasets are fetched programmatically by `data/loaders.py` and cached under
`data/cache/` (git-ignored). Nothing here is checked into version control.

| Loader | Dataset | Source | Used by |
|--------|---------|--------|---------|
| `load_card` | Card (1995) college proximity | Rdatasets (`wooldridge/card`) | 03 — instrumental variables |
| `load_mpdta` | County teen employment panel | `csdid`/`did` (Callaway & Sant'Anna) | 05 — staggered DiD |
| `load_lalonde` | LaLonde NSW job training | Rdatasets (`MatchIt/lalonde`) | 01/02 — matching, PSM |
| `load_organ_donations` | California organ-donor registry | `causaldata` | 04 — basic 2×2 DiD |
| `load_gov_transfers` | Honduras cash transfers | `causaldata` | 06 — regression discontinuity |
| `load_401k` | 401(k) eligibility & wealth | `doubleml` (Chernozhukov & Hansen) | 07/08/09 — DML, DML-IV, CATE |
| `load_germany_reunification` | West German GDP panel (Abadie 2015) | `pysyncon` mirror | 10 — synthetic control |

## Notes on provenance

- **Card** — from Card, D. (1995), "Using Geographic Variation in College
  Proximity to Estimate the Return to Schooling." Distributed with Wooldridge's
  *Introductory Econometrics* and mirrored by the public Rdatasets project.
- **mpdta** — minimum-wage / county teen-employment panel bundled with the R
  `did` package; this repo reads the copy already vendored in the sibling
  `csdid` project (`../csdid/data/mpdta.csv`), falling back to the upstream
  GitHub raw file if absent.
- **LaLonde** — the canonical NSW experimental + observational sample bundled
  with the R `MatchIt` package.

Rdatasets is a public mirror (GPL-compatible) maintained by Vincent
Arel-Bundock: <https://vincentarelbundock.github.io/Rdatasets/>.
