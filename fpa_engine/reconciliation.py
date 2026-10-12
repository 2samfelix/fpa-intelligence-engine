from __future__ import annotations

from decimal import Decimal
from enum import Enum

from fpa_engine.calculator import transitive_inputs
from fpa_engine.schemas import (
    BridgeDefinition,
    BridgeType,
    DisclosureStatus,
    FinancialFact,
    ReconciliationResult,
    ReconciliationStatus,
)
from fpa_engine.tolerance import bridge_tolerance

COMPATIBILITY_DIMENSIONS = ("unit", "scope", "price_basis", "basis")
ADDITIVE_DIMENSIONS = ("unit", "scope", "price_basis")


def _dimension(fact: FinancialFact, dim: str) -> str:
    value = getattr(fact, dim)
    return value.value if isinstance(value, Enum) else str(value)


def _mismatches(facts: list[FinancialFact], dimensions: tuple[str, ...]) -> list[str]:
    out = []
    for dim in dimensions:
        values = {_dimension(f, dim) for f in facts}
        if len(values) > 1:
            out.append(f"{dim}: {', '.join(sorted(values))}")
    return out


def _is_circular(bridge: BridgeDefinition, facts: dict[str, FinancialFact]) -> tuple[bool, list[str]]:
    ids = [bridge.starting_fact, bridge.ending_fact, *bridge.components.values()]
    reasons = [f"{i} is back-solved" for i in ids if facts[i].disclosure_status is DisclosureStatus.back_solved]
    component_ids = set(bridge.components.values())
    for endpoint in (bridge.starting_fact, bridge.ending_fact):
        overlap = transitive_inputs(endpoint, facts) & component_ids
        if overlap:
            reasons.append(f"{endpoint} is computed from bridge components {', '.join(sorted(overlap))}")
    for cid in component_ids:
        deps = transitive_inputs(cid, facts)
        if {bridge.starting_fact, bridge.ending_fact} & deps:
            reasons.append(f"component {cid} is computed from a bridge endpoint")
    return bool(reasons), reasons


def _base(bridge: BridgeDefinition, **kwargs) -> dict:
    return dict(
        id=bridge.id,
        bridge_name=bridge.name,
        period=bridge.period,
        scope=bridge.scope,
        bridge_type=bridge.bridge_type,
        unquantified_components=list(bridge.unquantified_components),
        **kwargs,
    )


def _evaluate_additive(bridge: BridgeDefinition, facts: dict[str, FinancialFact]) -> ReconciliationResult:
    input_ids = [bridge.starting_fact, bridge.ending_fact, *bridge.components.values()]
    unavailable = [i for i in input_ids if i not in facts or facts[i].value is None]
    if unavailable:
        return ReconciliationResult(**_base(
            bridge,
            component_fact_ids=dict(bridge.components),
            status=ReconciliationStatus.UNRESOLVED,
            reason=f"input unavailable: {', '.join(unavailable)}; residual cannot be computed",
            input_fact_ids=input_ids,
        ))

    inputs = [facts[i] for i in input_ids]
    incompatible = _mismatches(inputs, ADDITIVE_DIMENSIONS)
    if incompatible:
        return ReconciliationResult(**_base(
            bridge,
            component_fact_ids=dict(bridge.components),
            status=ReconciliationStatus.NOT_APPLICABLE,
            reason="inputs are not on a comparable basis (" + "; ".join(incompatible) + "); not additive",
            input_fact_ids=input_ids,
        ))

    start = facts[bridge.starting_fact].value
    end = facts[bridge.ending_fact].value
    reported_change = end - start
    identified = {name: facts[fid].value for name, fid in bridge.components.items()}
    identified_change = sum(identified.values(), Decimal("0"))
    residual = reported_change - identified_change
    precisions = {f.id: f.precision for f in inputs if f.precision is not None}
    tolerance, precision_limited = bridge_tolerance(precisions, reported_change, bridge.footing)
    circular, circular_reasons = _is_circular(bridge, facts)

    if circular:
        status = ReconciliationStatus.UNRESOLVED
        reason = ("closure is circular (" + "; ".join(circular_reasons) + "); a back-solved input "
                  "makes the residual zero by construction, so it proves nothing")
    elif bridge.unquantified_components or bridge.completeness == "partial":
        status = ReconciliationStatus.UNRESOLVED
        missing = ", ".join(bridge.unquantified_components) or "undisclosed components"
        reason = f"bridge is only partly quantified ({missing}); unassigned residual {residual} left unassigned"
    elif abs(residual) <= tolerance.allowed:
        status = ReconciliationStatus.RECONCILED
        reason = f"residual {residual} within allowed {tolerance.allowed}"
    else:
        status = ReconciliationStatus.UNRESOLVED
        reason = f"residual {residual} exceeds allowed {tolerance.allowed}"

    return ReconciliationResult(**_base(
        bridge,
        starting_value=start,
        ending_value=end,
        reported_change=reported_change,
        component_fact_ids=dict(bridge.components),
        identified_components=identified,
        identified_change=identified_change,
        residual=residual,
        tolerance=tolerance,
        status=status,
        reason=reason,
        input_fact_ids=input_ids,
        circular=circular,
        precision_limited=precision_limited,
    ))


def _evaluate_non_additive(bridge: BridgeDefinition, facts: dict[str, FinancialFact]) -> ReconciliationResult:
    unavailable = [m for m in bridge.members if m not in facts or facts[m].value is None]
    if unavailable:
        return ReconciliationResult(**_base(
            bridge,
            status=ReconciliationStatus.UNRESOLVED,
            reason=f"member unavailable: {', '.join(unavailable)}; comparability cannot be assessed",
            input_fact_ids=list(bridge.members),
        ))
    mismatches = _mismatches([facts[m] for m in bridge.members], COMPATIBILITY_DIMENSIONS)
    if mismatches:
        return ReconciliationResult(**_base(
            bridge,
            status=ReconciliationStatus.NOT_APPLICABLE,
            reason=(f"rejected relation '{bridge.proposed_relation}': members are not on a common basis ("
                    + "; ".join(mismatches) + ")"),
            input_fact_ids=list(bridge.members),
        ))
    return ReconciliationResult(**_base(
        bridge,
        status=ReconciliationStatus.UNRESOLVED,
        reason="declared non-additive, but no basis mismatch was detected; requires review",
        input_fact_ids=list(bridge.members),
    ))


def evaluate_bridge(bridge: BridgeDefinition, facts: dict[str, FinancialFact]) -> ReconciliationResult:
    if bridge.bridge_type is BridgeType.non_additive_set:
        return _evaluate_non_additive(bridge, facts)
    return _evaluate_additive(bridge, facts)


def evaluate_all(bridges: tuple[BridgeDefinition, ...], facts: dict[str, FinancialFact]) -> list[ReconciliationResult]:
    return [evaluate_bridge(b, facts) for b in bridges]
