from __future__ import annotations

import inspect
from decimal import Decimal

import pytest

from fpa_engine import calculator
from fpa_engine.calculator import REGISTRY, calculate_all
from fpa_engine.loader import load_benchmark
from fpa_engine.schemas import CalcStatus, Origin
from fpa_engine.tolerance import WORKBOOK_CACHE_TOLERANCE, matches_display, matches_workbook_cache

DERIVED_IDS = [f"D{i:02d}" for i in range(1, 26)]


@pytest.mark.parametrize("fact_id", DERIVED_IDS)
def test_derived_fact_recomputes(result, expected, fact_id):
    calc = result.calculations[fact_id]
    exp = expected["derived"][fact_id]
    assert calc.status is CalcStatus.computed, calc.reason
    if exp["workbook_cached"] is not None:
        assert matches_workbook_cache(calc.value, Decimal(exp["workbook_cached"]), Decimal(exp["workbook_scale"])), (
            f"{fact_id}: {calc.value} vs cached {exp['workbook_cached']} x {exp['workbook_scale']}"
        )
    else:
        assert abs(calc.value - Decimal(exp["expected_value"])) <= WORKBOOK_CACHE_TOLERANCE


DISPLAYED_IDS = [fid for fid, exp in load_benchmark().expected["derived"].items() if exp["display"] is not None]


@pytest.mark.parametrize("fact_id", DISPLAYED_IDS)
def test_derived_display_rounding(result, expected, fact_id):
    exp = expected["derived"][fact_id]
    assert matches_display(result.calculations[fact_id].value, Decimal(exp["display"]), Decimal(exp["display_unit"]))


@pytest.mark.parametrize("fact_id", DERIVED_IDS)
def test_derived_function_and_ordered_inputs(benchmark, expected, fact_id):
    fact = benchmark.fact_map[fact_id]
    exp = expected["derived"][fact_id]
    assert fact.formula == exp["function"]
    assert fact.formula in REGISTRY
    assert fact.inputs == exp["inputs"]


def test_d09_keeps_rounded_margin_basis(result, benchmark):
    d09 = result.facts["D09"]
    assert d09.value == Decimal("320")
    assert d09.basis == "rounded_disclosed_margin_difference"
    assert any("323.2" in note for note in d09.comparability_notes)
    facts = benchmark.fact_map
    raw_alternative = (facts["S13"].value / facts["S01"].value - facts["S14"].value / facts["S02"].value) * 10000
    assert raw_alternative.quantize(Decimal("0.1")) == Decimal("323.2")
    assert d09.value != raw_alternative


def test_d15_is_derived_not_disclosed(result):
    d15 = result.facts["D15"]
    assert d15.origin is Origin.derived
    assert d15.source_ids == []
    assert d15.value == Decimal("283.764")


@pytest.mark.parametrize("fact_id, half_width", [
    ("D07", "0.5"), ("D08", "0.5"), ("D09", "10"), ("D10", "10"), ("D23", "10"), ("D01", "1"),
])
def test_rounding_uncertainty_propagates(result, expected, fact_id, half_width):
    assert result.calculations[fact_id].uncertainty == Decimal(half_width)
    if "uncertainty" in expected["derived"][fact_id]:
        assert expected["derived"][fact_id]["uncertainty"] == half_width


def test_calculation_is_deterministic(benchmark):
    first = calculate_all(benchmark.fact_map)
    second = calculate_all(benchmark.fact_map)
    assert {k: v.value for k, v in first.items()} == {k: v.value for k, v in second.items()}


def test_values_are_decimal_not_float(result):
    for fid in DERIVED_IDS:
        assert isinstance(result.calculations[fid].value, Decimal)


def test_registry_has_no_free_form_evaluation():
    source = inspect.getsource(calculator)
    assert "eval(" not in source and "exec(" not in source


def test_growth_rate_guards_near_zero_and_sign_change(benchmark):
    facts = benchmark.fact_map
    zero_prior = facts["S02"].model_copy(update={"value": Decimal("0")})
    out = calculate_all({**facts, "S02": zero_prior})
    assert out["D03"].status is CalcStatus.not_computable
    assert "denominator" in out["D03"].reason

    negative_prior = facts["S02"].model_copy(update={"value": Decimal("-10")})
    out = calculate_all({**facts, "S02": negative_prior})
    assert out["D03"].status is CalcStatus.not_computable
    assert "sign change" in out["D03"].reason
