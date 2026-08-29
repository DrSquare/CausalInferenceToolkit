"""Thin, lazily-imported bridge to R via rpy2.

Methods in the deck that have no mature Python implementation (e.g. optimal
matching with ``MatchIt``/``optmatch``, or ``grf`` causal forests) are run
through R. Keeping the rpy2 machinery behind this module means the Python-only
examples never import rpy2, so they work without a local R installation.

On import this module best-effort configures ``R_HOME`` and ``PATH`` by locating
a local R installation, so callers don't have to set environment variables by
hand (particularly on Windows, where rpy2 needs R's ``bin`` dir on ``PATH`` to
load ``R.dll``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


def _configure_r_home() -> None:
    """Best-effort discovery of a local R install; set R_HOME + PATH."""
    if os.environ.get("R_HOME"):
        _prepend_r_bin(os.environ["R_HOME"])
        return

    candidates = []
    if sys.platform.startswith("win"):
        for base in (r"C:\Program Files\R", r"C:\Program Files (x86)\R"):
            p = Path(base)
            if p.is_dir():
                # Highest version number first (e.g. R-4.6.1 before R-4.2.2).
                candidates += sorted(p.glob("R-*"), reverse=True)
    else:
        candidates += [Path("/usr/lib/R"), Path("/usr/local/lib/R"),
                       Path("/Library/Frameworks/R.framework/Resources")]

    for c in candidates:
        if c.is_dir():
            os.environ["R_HOME"] = str(c)
            _prepend_r_bin(str(c))
            return


def _prepend_r_bin(r_home: str) -> None:
    """Put R's architecture bin dir on PATH so the shared library resolves."""
    for sub in (Path(r_home) / "bin" / "x64", Path(r_home) / "bin"):
        if sub.is_dir():
            if str(sub) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(sub) + os.pathsep + os.environ.get("PATH", "")
            break


_configure_r_home()


def r_available() -> bool:
    """Return True if rpy2 (and therefore an R runtime) can be imported."""
    try:
        import rpy2.robjects  # noqa: F401
    except Exception:
        return False
    return True


def require_r() -> None:
    """Raise a helpful error if the R bridge is unavailable."""
    if not r_available():
        raise RuntimeError(
            "This example needs R + rpy2. Install with:\n"
            '    python -m pip install -e ".[r]"\n'
            "and ensure a working R is on PATH."
        )


def ensure_packages(packages: Iterable[str]) -> None:
    """Install any missing CRAN packages into the active R library."""
    require_r()
    from rpy2.robjects.packages import importr, isinstalled

    utils = importr("utils")
    to_install = [p for p in packages if not isinstalled(p)]
    if to_install:
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(_r_strvector(to_install))


def to_r(df):
    """Convert a pandas DataFrame to an R data.frame."""
    require_r()
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter

    with localconverter(default_converter + pandas2ri.converter):
        return pandas2ri.py2rpy(df)


def to_pandas(r_df):
    """Convert an R data.frame back to a pandas DataFrame."""
    require_r()
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter

    with localconverter(default_converter + pandas2ri.converter):
        return pandas2ri.rpy2py(r_df)


def _r_strvector(values):
    from rpy2.robjects import StrVector

    return StrVector(list(values))
