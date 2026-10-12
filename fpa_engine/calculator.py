from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Callable

from fpa_engine.schemas import CalcStatus, CalculatedValue, FinancialFact, Origin, Unit
from fpa_engine.tolerance import BPS_PER_PCT_POINT, NotComputable, guard_denominator, half_unit, linear_uncertainty

HUNDRED = Decimal("100")
BPS_PER_UNIT = Decimal("10000")
DECIMAL_PRECISION = 34
GROWTH_BLOCKED_METRIC_TOKENS = ("dealer_inventory_change",)


@dataclass(frozen=True)
class Formula:
    name: str
    func: Callable[..., Decimal]
    input_units: tuple[Unit, ...]
    output_unit: Unit
    variadic_tail: bool = False
    same_scope: bool = False
    same_metric: bool = False
    denominator_arg: int | None = None
    growth: bool = False
    uncertainty_scale: Decimal | None = None


def _absolute_change(end: Decimal, start: Decimal) -> Decimal:
    return end - start


def _growth_rate(current: Decimal, prior: Decimal) -> Decimal:
    return (current / prior - 1) * HUNDRED


def _price_adjusted_growth(change: Decimal, price: Decimal, prior: Decimal) -> Decimal:
    return (change - price) / prior * HUNDRED


def _point_gap(a: Decimal, b: Decimal) -> Decimal:
    return a - b


def _pct_point_difference_bps(a: Decimal, b: Decimal) -> Decimal:
    return (a - b) * BPS_PER_PCT_POINT


def _ratio_pct(numerator: Decimal, denominator: Decimal) -> Decimal:
    return numerator / denominator * HUNDRED


def _drag_from_bps(bps: Decimal, base: Decimal) -> Decimal:
    return bps / BPS_PER_UNIT * base


def _ex_tariff_margin_add_back(profit: Decimal, drag: Decimal, sales: Decimal) -> Decimal:
    return (profit + drag) / sales * HUNDRED


def _ex_tariff_margin_from_cost(profit: Decimal, signed_cost: Decimal, sales: Decimal) -> Decimal:
    return (profit - signed_cost) / sales * HUNDRED


def _implied_stocking_swing(change: Decimal, price: Decimal, retail_pct: Decimal, prior: Decimal) -> Decimal:
    return (change - price) - retail_pct / HUNDRED * prior


def _back_solved_prior(end: Decimal, *components: Decimal) -> Decimal:
    return end - sum(components, Decimal("0"))


_USD, _PCT, _PTS, _BPS = Unit.usd_m, Unit.pct, Unit.pts, Unit.bps

REGISTRY: dict[str, Formula] = {
    f.name: f
    for f in (
        Formula("absolute_change", _absolute_change, (_USD, _USD), _USD,
                same_scope=True, same_metric=True, uncertainty_scale=Decimal("1")),
        Formula("growth_rate", _growth_rate, (_USD, _USD), _PCT,
                same_scope=True, same_metric=True, denominator_arg=1, growth=True),
        Formula("price_adjusted_growth", _price_adjusted_growth, (_USD, _USD, _USD), _PCT,
                same_scope=True, denominator_arg=2),
        Formula("point_gap", _point_gap, (_PCT, _PCT), _PTS, uncertainty_scale=Decimal("1")),
        Formula("margin_change_bps", _pct_point_difference_bps, (_PCT, _PCT), _BPS,
                same_scope=True, same_metric=True, uncertainty_scale=BPS_PER_PCT_POINT),
        Formula("tariff_distortion_bps", _pct_point_difference_bps, (_PCT, _PCT), _BPS,
                same_scope=True, uncertainty_scale=BPS_PER_PCT_POINT),
        Formula("ratio_pct", _ratio_pct, (_USD, _USD), _PCT, denominator_arg=1),
        Formula("drag_from_bps", _drag_from_bps, (_BPS, _USD), _USD, same_scope=True),
        Formula("ex_tariff_margin_add_back", _ex_tariff_margin_add_back, (_USD, _USD, _USD), _PCT,
                same_scope=True, denominator_arg=2),
        Formula("ex_tariff_margin_from_cost", _ex_tariff_margin_from_cost, (_USD, _USD, _USD), _PCT,
                same_scope=True, denominator_arg=2),
        Formula("implied_stocking_swing", _implied_stocking_swing, (_USD, _USD, _PCT, _USD), _USD),
        Formula("back_solved_prior", _back_solved_prior, (_USD, _USD), _USD,
                variadic_tail=True, same_scope=True),
    )
}


def expected_input_units(formula: Formula, n_inputs: int) -> tuple[Unit, ...] | None:
    if formula.variadic_tail:
        if n_inputs < len(formula.input_units):
            return None
        tail = formula.input_units[-1]
        return formula.input_units[:-1] + (tail,) * (n_inputs - len(formula.input_units) + 1)
    return formula.input_units if n_inputs == len(formula.input_units) else None


def dependency_order(facts: dict[str, FinancialFact]) -> tuple[list[str], set[str]]:
    """Topological order of derived facts, plus the IDs caught in a cycle."""
    derived = {fid: f for fid, f in facts.items() if f.origin is Origin.derived}
    order: list[str] = []
    state: dict[str, str] = {}
    cyclic: set[str] = set()

    def visit(fid: str, stack: list[str]) -> None:
        if state.get(fid) == "done":
            return
        if state.get(fid) == "active":
            cyclic.update(stack[stack.index(fid):])
            return
        state[fid] = "active"
        stack.append(fid)
        for dep in derived[fid].inputs:
            if dep in derived:
                visit(dep, stack)
        stack.pop()
        state[fid] = "done"
        order.append(fid)

    for fid in derived:
        visit(fid, [])
    return [fid for fid in order if fid not in cyclic], cyclic


def transitive_inputs(fact_id: str, facts: dict[str, FinancialFact]) -> set[str]:
    seen: set[str] = set()
    pending = list(facts[fact_id].inputs) if fact_id in facts else []
    while pending:
        dep = pending.pop()
        if dep in seen:
            continue
        seen.add(dep)
        if dep in facts:
            pending.extend(facts[dep].inputs)
    return seen


def _guard(formula: Formula, operands: list[FinancialFact], values: list[Decimal]) -> None:
    if formula.growth:
        blocked = [f.id for f in operands if any(t in f.metric for t in GROWTH_BLOCKED_METRIC_TOKENS)]
        if blocked:
            raise NotComputable(
                f"percentage growth is not meaningful for dealer-inventory changes ({', '.join(blocked)}); "
                "use the difference instead"
            )
        if values[0] * values[1] < 0:
            raise NotComputable("percentage growth across a sign change is not meaningful")
    if formula.denominator_arg is not None:
        denominator = operands[formula.denominator_arg]
        guard_denominator(values[formula.denominator_arg], denominator.precision or Decimal("1"))


def evaluate(
    fact: FinancialFact,
    operands: list[FinancialFact],
    operand_uncertainty: dict[str, Decimal | None],
) -> CalculatedValue:
    formula = REGISTRY.get(fact.formula or "")
    if formula is None:
        return CalculatedValue(fact_id=fact.id, status=CalcStatus.not_computable,
                               reason=f"formula '{fact.formula}' is not registered")
    values = [op.value for op in operands]
    if any(v is None for v in values):
        missing = [op.id for op in operands if op.value is None]
        return CalculatedValue(fact_id=fact.id, status=CalcStatus.unavailable,
                               reason=f"input value unavailable: {', '.join(missing)}")
    try:
        _guard(formula, operands, values)
        with localcontext() as ctx:
            ctx.prec = DECIMAL_PRECISION
            result = formula.func(*values)
    except NotComputable as exc:
        return CalculatedValue(fact_id=fact.id, status=CalcStatus.not_computable, reason=str(exc))

    uncertainty = None
    if formula.uncertainty_scale is not None:
        half_widths = []
        for op in operands:
            if op.origin is Origin.sourced:
                half_widths.append(half_unit(op.precision))
            else:
                half_widths.append(operand_uncertainty.get(op.id) or Decimal("0"))
        uncertainty = linear_uncertainty(half_widths, formula.uncertainty_scale)
    return CalculatedValue(fact_id=fact.id, status=CalcStatus.computed, value=result, uncertainty=uncertainty)


def calculate_all(facts: dict[str, FinancialFact]) -> dict[str, CalculatedValue]:
    order, cyclic = dependency_order(facts)
    results: dict[str, CalculatedValue] = {
        fid: CalculatedValue(fact_id=fid, status=CalcStatus.not_computable, reason="circular formula dependency")
        for fid in cyclic
    }
    resolved: dict[str, FinancialFact] = {fid: f for fid, f in facts.items() if f.origin is Origin.sourced}
    uncertainty: dict[str, Decimal | None] = {}

    for fid in order:
        fact = facts[fid]
        missing = [i for i in fact.inputs if i not in facts]
        blocked = [i for i in fact.inputs if i in results and results[i].status is not CalcStatus.computed]
        if missing or blocked:
            reasons = []
            if missing:
                reasons.append(f"missing input: {', '.join(missing)}")
            if blocked:
                reasons.append(f"input not computed: {', '.join(blocked)}")
            results[fid] = CalculatedValue(fact_id=fid, status=CalcStatus.unavailable, reason="; ".join(reasons))
            continue
        result = evaluate(fact, [resolved[i] for i in fact.inputs], uncertainty)
        results[fid] = result
        if result.status is CalcStatus.computed:
            resolved[fid] = fact.model_copy(update={"value": result.value})
            uncertainty[fid] = result.uncertainty
    return results


def materialize(facts: dict[str, FinancialFact], results: dict[str, CalculatedValue]) -> dict[str, FinancialFact]:
    """Facts with derived values filled in; unavailable derived facts keep value=None."""
    out = dict(facts)
    for fid, result in results.items():
        if fid in out:
            out[fid] = out[fid].model_copy(update={"value": result.value})
    return out
