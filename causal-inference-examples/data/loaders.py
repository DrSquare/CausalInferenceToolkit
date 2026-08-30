"""Cached loaders for the public datasets used across the examples.

Every loader downloads from a stable public source on first use, caches the CSV
under ``data/cache/`` (git-ignored), and returns a tidy :class:`pandas.DataFrame`.
Provenance and licenses are documented in ``data/README.md``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# csdid lives as a sibling project at the monorepo root; its bundled copy of the
# Callaway & Sant'Anna `mpdta` dataset is the canonical source for example 05.
_CSDID_DATA = Path(__file__).resolve().parents[3] / "csdid" / "data"


def _cached(name: str, url: str) -> pd.DataFrame:
    """Return a dataframe for ``url``, caching the raw CSV locally by ``name``."""
    path = CACHE_DIR / name
    if not path.exists():
        pd.read_csv(url).to_csv(path, index=False)
    return pd.read_csv(path)


def load_card() -> pd.DataFrame:
    """Card (1995) proximity-to-college data — instrumental variables example.

    Instrument ``nearc4`` (grew up near a 4-year college), treatment ``educ``
    (years of schooling), outcome ``lwage`` (log wage).
    """
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/card.csv"
    return _cached("card.csv", url)


def load_mpdta() -> pd.DataFrame:
    """County teen-employment panel (Callaway & Sant'Anna) — staggered DiD.

    Columns: ``year``, ``countyreal`` (unit id), ``lemp`` (log employment),
    ``lpop`` (log population, a covariate), ``first.treat`` (first treated year;
    0 = never treated), ``treat``.
    """
    local = _CSDID_DATA / "mpdta.csv"
    if local.exists():
        return pd.read_csv(local)
    url = "https://raw.githubusercontent.com/bcallaway11/did/master/data-raw/mpdta.csv"
    return _cached("mpdta.csv", url)


def load_lalonde() -> pd.DataFrame:
    """LaLonde NSW job-training data — matching / propensity-score examples."""
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv"
    return _cached("lalonde.csv", url)


def load_organ_donations() -> pd.DataFrame:
    """California organ-donor registration panel — basic 2x2 DiD example.

    From the ``causaldata`` package. California (treated) switched to an active-
    choice online registry in Q3 2011; other states are controls. Columns:
    ``State``, ``Quarter``, ``Rate`` (donor registration rate), ``Quarter_Num``.
    """
    from causaldata import organ_donations

    return organ_donations.load_pandas().data


def load_gov_transfers() -> pd.DataFrame:
    """Honduras conditional-cash-transfer data — regression discontinuity.

    From ``causaldata`` (Manacorda, Miguel & Vigorito). Running variable
    ``Income_Centered`` (eligibility cutoff at 0; households below 0 are
    eligible), outcomes ``Support`` and ``Education``.
    """
    from causaldata import gov_transfers

    return gov_transfers.load_pandas().data


def load_401k() -> pd.DataFrame:
    """401(k) eligibility / participation and wealth — DML / DML-IV / CATE.

    From ``doubleml`` (Chernozhukov & Hansen). Outcome ``net_tfa`` (net
    financial assets); ``e401`` = eligibility (used as treatment for DML and as
    the instrument for DML-IV); ``p401`` = participation (endogenous treatment);
    plus demographic/financial covariates.
    """
    from doubleml.datasets import fetch_401K

    return fetch_401K("DataFrame")


def load_germany_reunification() -> pd.DataFrame:
    """West German GDP panel (Abadie 2015) — synthetic control example.

    Long-format panel of per-capita GDP for OECD countries; the treated unit is
    West Germany (reunification in 1990). Mirrored from the pysyncon examples.
    """
    url = "https://raw.githubusercontent.com/sdfordham/pysyncon/main/data/germany.csv"
    return _cached("germany.csv", url)
