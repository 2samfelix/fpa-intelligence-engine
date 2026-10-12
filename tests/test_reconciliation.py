from __future__ import annotations

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fpa_engine.calculator import calculate_all, materialize
from fpa_engine.pipeline import main
from fpa_engine.reconciliation import evaluate_bridge
from fpa_engine.schemas import (
    BridgeDefinition,
    BridgeType,
    FinancialFact,
    ReconciliationResult,
    ReconciliationStatus,
)
from fpa_engine.tolerance import bridge_tolerance

BRIDGE_IDS = ["R01", "R02", "R03", "R04", "R05"]
NON_ADDITIVE_IDS = [f"NA0{i}" for i in range(1, 9)]


def _usd(fid: str, value: str, precision: str = "1", metric: str = "m") -> FinancialFact:
    return FinancialFact(
        id=fid, metric=metric, value=Decimal(value), unit="usd_m", period="Q2 2026", period_type="three_months",
        origin="sourced", disclosure_status="disclosed", source_ids=["X"], basis="as_reported",
        precision=Decimal(precision), scope="Consolidated", price_basis="current", sign_convention="level",
    )


@pytest.mark.parametrize("bridge_id", BRIDGE_IDS)
def test_additive_bridge_matches_audit(recon, expected, bridge_id):
    r, exp = recon[bridge_id], expected["reconciliations"][bridge_id]
    assert r.status.value == exp["status"]
    assert r.residual == Decimal(exp["residual"])
    assert r.tolerance.allowed == Decimal(exp["allowed"])
    assert r.circular is exp["circular"]


def test_valid_sales_bridges_reconcile_exactly(recon):
    for rid in ("R01", "R02"):
        r = recon[rid]
        assert r.status is ReconciliationStatus.RECONCILED
        assert r.reported_change == r.identified_change
        assert r.residual == 0
        assert not r.unquantified_components


def test_profit_bridges_preserve_exact_unassigned_residual(recon):
    assert recon["R03"].reported_change == 511 and recon["R03"].identified_change == 499
    assert recon["R04"].identified_change == 0 and recon["R04"].residual == 703
    assert recon["R05"].reported_change == -144 and recon["R05"].identified_change == -158
    for rid in ("R03", "R04", "R05"):
        assert recon[rid].status is ReconciliationStatus.UNRESOLVED
        assert recon[rid].unquantified_components


@pytest.mark.parametrize("set_id", NON_ADDITIVE_IDS)
def test_non_additive_sets_are_not_applicable(recon, expected, set_id):
    r = recon[set_id]
    assert r.status is ReconciliationStatus.NOT_APPLICABLE
    assert expected["reconciliations"][set_id]["status"] == "NOT_APPLICABLE"
    assert r.residual is None
    assert "not on a common basis" in r.reason


def test_demand_quality_triad_rejected_on_basis_grounds(recon):
    reason = recon["NA01"].reason
    assert "unit" in reason and "price_basis" in reason and "scope" in reason


def test_reconciled_results_are_never_circular(result):
    for r in result.reconciliations:
        if r.status is ReconciliationStatus.RECONCILED:
            assert not r.circular


def test_back_solved_endpoint_makes_bridge_circular(benchmark):
    back_solved = FinancialFact(
        id="S02", metric="ci_segment_sales", unit="usd_m", period="Q2 2025", period_type="three_months",
        origin="derived", disclosure_status="back_solved", formula="back_solved_prior",
        inputs=["S01", "S03", "S04", "S05", "S06"], basis="back_solved", scope="CI", price_basis="current",
        sign_convention="level",
    )
    facts = benchmark.fact_map | {"S02": back_solved}
    facts = materialize(facts, calculate_all(facts))
    r = evaluate_bridge(next(b for b in benchmark.bridges if b.id == "R01"), facts)
    assert r.residual == 0
    assert r.circular is True
    assert r.status is ReconciliationStatus.UNRESOLVED
    assert "circular" in r.reason


def test_tolerance_rule_matches_gate_a_formula():
    precisions = {f"i{n}": Decimal("1") for n in range(6)}
    tol, limited = bridge_tolerance(precisions, Decimal("2156"), footing="none")
    assert tol.rounding_floor == Decimal("3.0")
    assert tol.cap == Decimal("21.56")
    assert tol.allowed == Decimal("3.0")
    assert not limited
    footed, _ = bridge_tolerance(precisions, Decimal("2156"), footing="exact")
    assert footed.allowed == 0


def test_tolerance_cap_floor_for_small_changes():
    tol, _ = bridge_tolerance({"a": Decimal("1"), "b": Decimal("1")}, Decimal("20"), footing="none")
    assert tol.cap == Decimal("1")
    assert tol.allowed == Decimal("1")


def test_precision_limited_reconciliation_synthetic():
    # Tolerance fixture only (NOT part of the v1 benchmark): Consolidated Q2 2026 values from
    # Consolidated!I5-I11, using the disclosed "$3.1 billion" volume instead of the back-solved 3,113.
    facts = {f.id: f for f in (
        _usd("C0", "16569"), _usd("C1", "20543"), _usd("CV", "3100", precision="100"),
        _usd("CP", "595"), _usd("CF", "199"), _usd("CO", "67"),
    )}
    bridge = BridgeDefinition(
        id="TOL1", name="Consolidated sales bridge", period="Q2 2026", scope="Consolidated",
        bridge_type="company_disclosed_bridge", starting_fact="C0", ending_fact="C1",
        components={"volume": "CV", "price": "CP", "currency": "CF", "other": "CO"},
    )
    r = evaluate_bridge(bridge, facts)
    assert r.residual == Decimal("13")
    assert r.tolerance.rounding_floor == Decimal("52.5")
    assert r.tolerance.allowed == Decimal("39.74")
    assert r.status is ReconciliationStatus.RECONCILED
    assert r.precision_limited is True


def test_declared_non_additive_set_with_compatible_members_is_not_auto_rejected(benchmark):
    facts = benchmark.fact_map
    bridge = BridgeDefinition(
        id="NAX", name="mislabelled set", period="Q2 2026", scope="CI", bridge_type="non_additive_set",
        members=["S03", "S05"], proposed_relation="volume + currency",
    )
    r = evaluate_bridge(bridge, facts)
    assert r.status is ReconciliationStatus.UNRESOLVED
    assert "requires review" in r.reason


def test_result_schema_forbids_invalid_states():
    base = dict(id="X", bridge_name="x", period="p", scope="s", bridge_type=BridgeType.non_additive_set, reason="r")
    with pytest.raises(ValidationError):
        ReconciliationResult(**base, status=ReconciliationStatus.NOT_APPLICABLE, residual=Decimal("0"))
    with pytest.raises(ValidationError):
        ReconciliationResult(**base, status=ReconciliationStatus.RECONCILED, circular=True)


def test_regression_dataset_not_encoded(expected):
    regression = expected["regression"]
    assert regression["encoded"] is False
    assert regression["gate_a_expectation"] == {"independently_reconciled": 28, "precision_limited": 1,
                                                "circular_unresolved": 3}


def test_pipeline_cli_summary(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Facts loaded: 40 sourced, 25 derived definitions" in out
    assert "Validation: PASS" in out
    assert "R01 CI sales bridge Q2 2026: RECONCILED" in out
    assert "R03 CI segment profit bridge Q1 2026: UNRESOLVED" in out
    assert "Residual: +$12M" in out
    assert "NA01 Demand-quality triad Q2 2026: NOT_APPLICABLE" in out


def test_pipeline_cli_json(capsys):
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    statuses = {r["id"]: r["status"] for r in payload["reconciliations"]}
    assert statuses["R01"] == "RECONCILED" and statuses["R04"] == "UNRESOLVED"
    assert payload["validation"]["ok"] is True
