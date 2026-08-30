"""Compatibility shims so Microsoft's SparseSC runs on modern SciPy/scikit-learn.

SparseSC 0.2.0 targets older SciPy/scikit-learn and breaks on current versions:

1. ``scipy.optimize.linesearch.LineSearchWarning`` was made private
   (moved to ``scipy.optimize._linesearch``) in SciPy >= 1.11.
2. The ``normalize`` argument to ``Lasso``/``MultiTaskLasso``/``LassoCV``/
   ``MultiTaskLassoCV`` was removed in scikit-learn 1.2.
3. ``RidgeCV(store_cv_values=...)`` was renamed to ``store_cv_results`` in
   scikit-learn 1.5.

``patch()`` is idempotent and must be called *before* SparseSC does its fitting.
It re-binds the affected names inside SparseSC's own modules to thin factory
functions that drop/translate the removed keywords, so no fork of SparseSC is
needed. This keeps the rest of the toolkit on current scikit-learn (which
econml/doubleml require).
"""
from __future__ import annotations

_PATCHED = False


def _drop_normalize(real):
    def factory(*args, normalize=None, **kwargs):  # noqa: ARG001 - intentionally ignored
        return real(*args, **kwargs)

    return factory


def _rename_ridge(real):
    """Return a RidgeCV subclass that accepts the old ``store_cv_values`` kwarg
    and exposes the old ``cv_values_`` attribute (both renamed in sklearn 1.5)."""

    class _CompatRidgeCV(real):
        @property
        def cv_values_(self):
            return self.cv_results_

    def factory(*args, store_cv_values=None, **kwargs):
        if store_cv_values is not None:
            kwargs["store_cv_results"] = store_cv_values
        return _CompatRidgeCV(*args, **kwargs)

    return factory


def patch() -> None:
    """Apply all SparseSC compatibility shims (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return

    # (1) SciPy: expose LineSearchWarning where SparseSC's optimizers import it.
    import scipy.optimize.linesearch as _ls
    from scipy.optimize._linesearch import LineSearchWarning as _LSW

    if not hasattr(_ls, "LineSearchWarning"):
        _ls.LineSearchWarning = _LSW

    # Import after the SciPy shim so SparseSC's own imports succeed.
    import sklearn.linear_model as _sklm
    import SparseSC.utils.match_space as _match_space
    from sklearn.linear_model import (
        Lasso,
        LassoCV,
        MultiTaskLasso,
        MultiTaskLassoCV,
        RidgeCV,
    )

    # (2) scikit-learn: strip the removed ``normalize`` keyword. match_space
    #     binds these at import time, so patch the names on that module.
    _match_space.MultiTaskLassoCV = _drop_normalize(MultiTaskLassoCV)
    _match_space.MultiTaskLasso = _drop_normalize(MultiTaskLasso)
    _match_space.LassoCV = _drop_normalize(LassoCV)
    _match_space.Lasso = _drop_normalize(Lasso)

    # (3) scikit-learn: translate the renamed RidgeCV keyword. penalty_utils
    #     imports RidgeCV *inside* the function, so patch sklearn itself.
    _sklm.RidgeCV = _rename_ridge(RidgeCV)

    _PATCHED = True
