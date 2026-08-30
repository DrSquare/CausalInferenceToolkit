"""Smoke tests: each shipped example runs end-to-end on its public dataset."""
import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matching():
    if importlib.util.find_spec("rpy2") is None:
        pytest.skip("rpy2 not installed")
    sys.path.insert(0, str(EXAMPLES.parent / "src"))
    from citk_examples import _rutils

    if not _rutils.r_available():
        pytest.skip("R runtime not available to rpy2")
    from rpy2.robjects.packages import isinstalled

    if not isinstalled("MatchIt"):
        pytest.skip("R package MatchIt not installed")
    mod = _load("01_matching")
    res = mod.run()
    # Matching should improve covariate balance vs. the raw sample.
    assert res["nearest"]["smd_after"] <= res["nearest"]["smd_before"]
    assert res["caliper"]["smd_after"] < res["caliper"]["smd_before"]


def test_instrumental_variables():
    mod = _load("03_instrumental_variables")
    res = mod.run()
    # IV corrects OLS attenuation from ability bias; both should be sane returns.
    assert 0.0 < res["iv_beta"] < 0.5
    assert res["first_stage_f"] > 0


def test_staggered_did():
    if importlib.util.find_spec("csdid") is None:
        pytest.skip("csdid sibling package not installed")
    mod = _load("05_did_staggered")
    res = mod.run()
    assert isinstance(res["simple_att"], float)
    assert len(res["event_times"]) == len(res["dynamic_att"])


def test_propensity_iptw():
    mod = _load("02_propensity_iptw")
    res = mod.run()
    # Adjusting for observed confounders flips the naive sign toward a positive
    # training effect and shrinks covariate imbalance.
    assert res["iptw_att"] > res["naive"]
    assert res["max_smd_after"] < res["max_smd_before"]


def test_did_basic():
    if importlib.util.find_spec("causaldata") is None:
        pytest.skip("causaldata package not installed")
    mod = _load("04_did_basic")
    res = mod.run()
    # Regression interaction term must equal the manual 2x2 computation.
    assert abs(res["did"] - res["manual_did"]) < 1e-9


def test_regression_discontinuity():
    if importlib.util.find_spec("rdrobust") is None:
        pytest.skip("rdrobust not installed")
    mod = _load("06_regression_discontinuity")
    res = mod.run()
    assert 0.0 < res["bandwidth"]


def test_double_ml():
    if importlib.util.find_spec("doubleml") is None:
        pytest.skip("doubleml not installed")
    mod = _load("07_double_ml")
    res = mod.run()
    # Canonical 401(k) eligibility effect is a large positive number (~$9k).
    assert res["ate"] > 1000


def test_dml_iv():
    if importlib.util.find_spec("econml") is None:
        pytest.skip("econml not installed")
    mod = _load("08_dml_iv")
    res = mod.run()
    assert res["ci_low"] < res["late"] < res["ci_high"]


def test_heterogeneous_effects():
    if importlib.util.find_spec("econml") is None:
        pytest.skip("econml not installed")
    mod = _load("09_heterogeneous_effects")
    res = mod.run()
    # Effect should rise with income: top quartile CATE exceeds bottom quartile.
    cate = res["cate_by_income"]
    assert cate["Q4"] > cate["Q1"]


def test_synthetic_control():
    if importlib.util.find_spec("pysyncon") is None:
        pytest.skip("pysyncon not installed")
    mod = _load("10_synthetic_control")
    res = mod.run()
    # Reunification lowered West German GDP vs. its synthetic control.
    assert res["att"] < 0


def test_synthetic_control_sparsesc():
    if importlib.util.find_spec("SparseSC") is None:
        pytest.skip("SparseSC not installed")
    mod = _load("11_synthetic_control_sparsesc")
    res = mod.run()
    # ML-regularized synthetic control also finds a negative reunification
    # effect on West German GDP.
    assert res["att"] < 0
