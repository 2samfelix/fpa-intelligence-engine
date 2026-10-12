from __future__ import annotations

from decimal import Decimal

from fpa_engine.schemas import ToleranceResult

WORKBOOK_CACHE_TOLERANCE = Decimal("1e-9")
CAP_RATE = Decimal("0.01")
BPS_PER_PCT_POINT = Decimal("100")


class NotComputable(Exception):
    """Raised when a metric must not be computed (near-zero denominator, blocked growth)."""


def half_unit(precision: Decimal) -> Decimal:
    return precision / 2


def bridge_tolerance(
    input_precisions: dict[str, Decimal],
    reported_change: Decimal,
    footing: str,
) -> tuple[ToleranceResult, bool]:
    """Gate A rule: allowed = min(rounding_floor, cap).

    rounding_floor sums half the disclosure unit of every bridge input, except
    when the source presents the bridge as a single table that foots exactly;
    then any residual is a transcription error, so the floor is zero.
    cap = max(1% of |reported change|, 1 disclosure unit).
    Returns the tolerance and whether a coarser-than-finest input drove it.
    """
    if not input_precisions:
        raise ValueError("a bridge tolerance needs at least one input precision")
    disclosure_unit = min(input_precisions.values())
    if footing == "exact":
        rounding_floor = Decimal("0")
    else:
        rounding_floor = sum((half_unit(p) for p in input_precisions.values()), Decimal("0"))
    cap = max(abs(reported_change) * CAP_RATE, disclosure_unit)
    allowed = min(rounding_floor, cap)
    precision_limited = footing != "exact" and any(p > disclosure_unit for p in input_precisions.values())
    return (
        ToleranceResult(
            rounding_floor=rounding_floor,
            cap=cap,
            allowed=allowed,
            disclosure_unit=disclosure_unit,
            footing=footing,
            input_precisions=dict(input_precisions),
        ),
        precision_limited,
    )


def matches_workbook_cache(value: Decimal, cached: Decimal, scale: Decimal = Decimal("1")) -> bool:
    return abs(value - cached * scale) <= WORKBOOK_CACHE_TOLERANCE


def matches_display(value: Decimal, display: Decimal, display_unit: Decimal) -> bool:
    return abs(value - display) <= half_unit(display_unit)


def linear_uncertainty(input_half_widths: list[Decimal], scale: Decimal) -> Decimal:
    """Worst-case half-width of a sum/difference of rounded inputs, in output units.

    A difference of two margins disclosed to 0.1 pt, expressed in bps (scale 100),
    is good to +/-10 bps.
    """
    return sum(input_half_widths, Decimal("0")) * scale


def guard_denominator(denominator: Decimal, unit: Decimal) -> None:
    if abs(denominator) < unit:
        raise NotComputable(f"denominator {denominator} is below one disclosure unit ({unit})")
