from __future__ import annotations

from decimal import Decimal

import pytest

from fpa_engine.calculator import calculate_all
from fpa_engine.pipeline import ValidationFailed, run
from fpa_engine.schemas import CalcStatus, DisclosureStatus, ReconciliationStatus, Unit
from fpa_engine.validator import validate_inputs


def test_removing_s29_makes_tariff_metrics_unavailable(benchmark):
    degraded = benchmark.without_facts("S29")
    report = validate_inputs(degraded)
    assert any(i.code == "MISSING_INPUT" and i.subject_id == "D15" for i in report.errors)
    with pytest.raises(ValidationFailed):
        run(degraded)

    result = run(degraded, strict=False)
    assert result.calculations["D15"].status is CalcStatus.unavailable
    assert result.calculations["D16"].status is CalcStatus.unavailable
    assert result.facts["D15"].value is None and result.facts["D16"].value is None
    assert result.calculations["D17"].status is CalcStatus.computed
    na07 = next(r for r in result.reconciliations if r.id == "NA07")
    assert na07.status is ReconciliationStatus.UNRESOLVED and na07.residual is None


def test_removing_s23_removes_dependent_dealer_build_claims(benchmark):
    degraded = benchmark.without_facts("S23")
    result = run(degraded, strict=False)
    for fid in ("D13", "D14"):
        assert result.calculations[fid].status is CalcStatus.unavailable
        assert result.facts[fid].value is None
    lookup = degraded.lookup("ci_dealer_inventory_change", "CI", "Q1 2026")
    assert lookup.status == "not_in_benchmark" and lookup.fact is None
    for rid in ("NA02", "NA04", "NA05", "NA08"):
        r = next(x for x in result.reconciliations if x.id == rid)
        assert r.status is ReconciliationStatus.UNRESOLVED and r.residual is None


def test_corrupting_s03_breaks_r01(benchmark):
    s03 = benchmark.fact_map["S03"]
    corrupted = benchmark.with_fact(s03.model_copy(update={"value": s03.value + 1}))
    result = run(corrupted)
    r01 = next(r for r in result.reconciliations if r.id == "R01")
    assert r01.status is ReconciliationStatus.UNRESOLVED
    assert r01.residual == Decimal("-1")


def test_back_solved_value_fed_as_sourced_is_rejected(benchmark):
    s02 = benchmark.fact_map["S02"]
    disguised = benchmark.with_fact(s02.model_copy(update={"disclosure_status": DisclosureStatus.back_solved}))
    report = validate_inputs(disguised)
    assert any(i.code == "BACK_SOLVED_AS_SOURCED" and i.subject_id == "S02" for i in report.errors)
    with pytest.raises(ValidationFailed, match="BACK_SOLVED_AS_SOURCED"):
        run(disguised)


def test_unavailable_company_wide_q2_dealer_inventory_is_not_fabricated(benchmark, result):
    lookup = benchmark.lookup("dealer_inventory_change", "Company-wide", "Q2 2026")
    assert lookup.fact is None
    assert lookup.status == "not_collected"
    assert lookup.gap.id == "G01"
    assert not any(
        f.metric == "dealer_inventory_change" and f.period == "Q2 2026" for f in result.facts.values()
    )


def test_near_zero_denominator_is_not_computable(benchmark):
    s02 = benchmark.fact_map["S02"]
    facts = {**benchmark.fact_map, "S02": s02.model_copy(update={"value": Decimal("0.4")})}
    out = calculate_all(facts)
    for fid in ("D03", "D05"):
        assert out[fid].status is CalcStatus.not_computable
        assert out[fid].value is None
    assert out["D07"].status is CalcStatus.unavailable


def test_dealer_inventory_percentage_growth_is_blocked(benchmark):
    growth = benchmark.fact_map["D03"].model_copy(
        update={"id": "DX", "metric": "ci_dealer_inventory_growth", "inputs": ["S25", "S23"]}
    )
    out = calculate_all({**benchmark.fact_map, "DX": growth})
    assert out["DX"].status is CalcStatus.not_computable
    assert "dealer-inventory" in out["DX"].reason


def test_stale_343_bps_artifact_is_not_reproduced(benchmark, result, expected):
    q2_tariff = [f for f in result.facts.values() if f.metric == "ci_tariff_margin_impact" and f.period == "Q2 2026"]
    assert [(f.id, f.value) for f in q2_tariff] == [("S29", Decimal("340"))]
    rejected = {Decimal(r["value"]) for r in expected["rejected_values"]}
    bps_values = [f.value for f in result.facts.values() if f.unit is Unit.bps and f.value is not None]
    for value in bps_values:
        for bad in rejected:
            assert abs(value - bad) >= Decimal("0.5"), f"{value} reproduces rejected value {bad}"
    assert not any(f.period == "Q2 2026" and f.metric == "ci_tariff_distortion_bps" for f in result.facts.values())


def test_rejected_plug_values_never_appear(result):
    values = {f.value for f in result.facts.values() if f.value is not None}
    assert Decimal("3113") not in values
    assert Decimal("570") not in values
