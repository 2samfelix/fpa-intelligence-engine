"""Gate C4 step 1: the packet, not the model, owns public numeric presentation (packet v2)."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fpa_engine.brief import (
    BriefPacket,
    BriefStatement,
    PacketBuildError,
    build_brief_packet,
    format_display,
    load_policy,
    uncertainty_step,
)
from fpa_engine.brief_validator import validate_statements
from fpa_engine.schemas import Origin, Unit

T0 = datetime(2026, 10, 12, tzinfo=timezone.utc)
V1_HASH = "96fda5218790e13c4515bd63fc44bbd5e7c983fbbdf67b8ecbde312817a579bb"
STEP1_HASH = "6f5204299bf95d6ff49b58d60372ee7c896ff2717a5b3c45fcb8d735800105c6"
V2_HASH = "d726abc65d9d39f8077406bfcc3603596fd0dad36adb5e2421695d9f491409bc"
DISPLAY_CODES = {"FALSE_PRECISION", "APPROXIMATION_MARKER_MISSING", "UNSUPPORTED_NUMBER", "UNCITED_FACT_VALUE"}


def st(sid: str, kind: str, text: str, **ids) -> BriefStatement:
    return BriefStatement(id=sid, kind=kind, text=text, **ids)


def display_codes(packet, text: str, fact_id: str) -> set[str]:
    return validate_statements(packet, [st("A", "fact", text, fact_ids=[fact_id])]).codes & DISPLAY_CODES


# ------------------------------------------------------------ packet v2 ---

@pytest.mark.parametrize("fid, display, step", [
    ("D08", "~24 pts", "1"),
    ("D07", "~8 pts", "1"),
    ("D18", "~510 bps", "10"),
    ("D25", "~600 bps", "10"),
    ("D15", "~$284M", "1"),
    ("D13", "~25%", "5"),
])
def test_canonical_display_of_approximate_facts(packet, fid, display, step):
    f = packet.fact_map[fid]
    assert (f.display, f.display_step, f.approximate) == (display, Decimal(step), True)


def test_d13_carries_policy_wording(packet):
    d13 = packet.fact_map["D13"]
    assert d13.display_wording == "roughly a quarter" and d13.display == "~25%"


def test_d14_carries_policy_wording(packet):
    d14 = packet.fact_map["D14"]
    assert d14.display_wording == "roughly three quarters" and d14.display == "~75%"


def test_display_wording_defaults_to_the_display(packet):
    for f in packet.facts:
        if f.id not in ("D13", "D14"):
            assert f.display_wording == f.display, f.id


@pytest.mark.parametrize("fid, value", [
    ("D08", "24.26929012345679012345679012345679"),
    ("D07", "7.83844911147011308562197092084006"),
    ("D18", "509.0713587487781036168132942326490"),
    ("D15", "283.764"),
    ("D13", "26.66666666666666666666666666666667"),
])
def test_internal_values_are_unchanged(packet, result, fid, value):
    assert packet.fact_map[fid].value == Decimal(value) == result.facts[fid].value


def test_sourced_facts_keep_their_disclosed_display(packet):
    for f in packet.facts:
        if f.origin is Origin.sourced:
            assert not f.approximate and f.uncertainty is None, f.id
            assert f.display_step == Decimal(10) ** -f.max_decimals, f.id
    fm = packet.fact_map
    assert [fm[i].display for i in ("S23", "S25", "S29", "S17", "S21")] == ["$1,500M", "$400M", "340 bps", "23.3%", "7%"]


def test_precise_derived_facts_keep_their_display(packet):
    fm = packet.fact_map
    assert [fm[i].display for i in ("D01", "D03", "D04", "D11", "D16", "D17", "D24")] == [
        "$2,156M", "34.8%", "38.1%", "$703M", "26.7%", "26.5%", "20.9%"]
    assert not any(fm[i].approximate for i in ("D01", "D03", "D04", "D11", "D16", "D17", "D24"))


def test_display_step_is_never_finer_than_uncertainty(packet):
    for f in packet.facts:
        if f.uncertainty:
            assert f.display_step >= min(f.uncertainty, uncertainty_step(f.uncertainty)), f.id


def test_every_packet_display_passes_its_own_rules(packet):
    for f in packet.facts:
        assert display_codes(packet, f"The figure was {f.display} in the period.", f.id) == set(), f.id


@pytest.mark.parametrize("uncertainty, step", [("0.5", "1"), ("5", "10"), ("10", "10"), ("1.0", "1"), ("33", "100")])
def test_uncertainty_step_is_the_next_power_of_ten(uncertainty, step):
    assert uncertainty_step(Decimal(uncertainty)) == Decimal(step)


def test_format_display_marks_approximation():
    assert format_display(Decimal("-470"), Unit.bps, Decimal("10"), True) == "~-470 bps"
    assert format_display(Decimal("1258.12"), Unit.usd_m, Decimal("100"), True) == "~$1,300M"
    assert format_display(Decimal("34.83"), Unit.pct, Decimal("0.1")) == "34.8%"


def test_packet_version_is_2(packet):
    assert packet.packet_version == 2
    payload = packet.model_dump(mode="json")
    with pytest.raises(ValidationError):
        BriefPacket.model_validate({**payload, "packet_version": 1})


def test_packet_hash_changed_deterministically(benchmark, packet):
    later = build_brief_packet(benchmark, generated_at=datetime(2027, 1, 1, tzinfo=timezone.utc))
    assert packet.packet_hash == later.packet_hash == V2_HASH
    assert V2_HASH not in (V1_HASH, STEP1_HASH)


def test_display_policy_is_carried_in_the_packet(packet):
    assert {"~", "about", "roughly", "approximately"} <= set(packet.display_policy.approximation_markers)


@pytest.mark.parametrize("fid, override, message", [
    ("S29", {"approximate": True, "reason": "x"}, "sourced figures keep their disclosed display"),
    ("D13", {"display_step": "1", "reason": "x"}, "finer than the precision the inputs support"),
    ("D08", {"display_step": "0.1", "reason": "x"}, "finer than the precision the inputs support"),
    ("D08", {"approximate": False, "reason": "x"}, "cannot be marked exact"),
    ("D13", {"display_wording": "about 26.7%", "reason": "x"}, "may not contain digits"),
    ("D15", {"approximate": True}, "must give a reason"),
    ("Z99", {"approximate": True, "reason": "x"}, "not in the packet"),
])
def test_invalid_display_overrides_block_the_packet(benchmark, fid, override, message):
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    policy["display_policy"]["fact_overrides"][fid] = override
    with pytest.raises(PacketBuildError, match=message):
        build_brief_packet(benchmark, generated_at=T0, policy=policy)


def test_missing_display_policy_blocks_the_packet(benchmark):
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    del policy["display_policy"]
    with pytest.raises(PacketBuildError, match="DISPLAY_POLICY_INVALID"):
        build_brief_packet(benchmark, generated_at=T0, policy=policy)


# -------------------------------------------------------------- validator ---

@pytest.mark.parametrize("text, fid", [
    ("The Q1 2026 adjusted gap was 24.3 pts.", "D08"),
    ("The Q2 2026 adjusted gap was 7.8 pts.", "D07"),
    ("The Q2 2026 adjusted gap was ~7.8 pts.", "D07"),
    ("The Q1 2026 adjusted gap was about 24.0 pts.", "D08"),
    ("The Q1 2026 distortion was 509 bps.", "D18"),
    ("The Q1 2026 distortion was about 509 bps.", "D18"),
    ("The Q1 2026 distortion was about 5.09 percentage points.", "D18"),
    ("The Q2 2026 build was 26.7% of the Q1 2026 build.", "D13"),
    ("The Q2 2026 build was about 27% of the Q1 2026 build.", "D13"),
    ("The CI build was 75.0% of the company-wide build.", "D14"),
    ("The implied Q2 2026 swing was about $485M.", "D20"),
    ("The implied Q1 2026 swing was about $1,258M.", "D21"),
    ("Reported growth was 34.83% in Q2 2026.", "D03"),
])
def test_over_precise_renderings_are_rejected(packet, text, fid):
    assert display_codes(packet, text, fid) == {"FALSE_PRECISION"}


@pytest.mark.parametrize("text, fid", [
    ("The Q1 2026 adjusted gap was 24 pts.", "D08"),
    ("The Q2 2026 adjusted gap was 8 pts.", "D07"),
    ("The Q1 2026 distortion was 510 bps.", "D18"),
    ("The Q2 2026 tariff drag was $284M.", "D15"),
    ("The Q2 2026 build was 25% of the Q1 2026 build.", "D13"),
    ("CI margin fell 470 bps in Q4 2025.", "D23"),
])
def test_missing_approximation_marker_is_rejected(packet, text, fid):
    assert display_codes(packet, text, fid) == {"APPROXIMATION_MARKER_MISSING"}


@pytest.mark.parametrize("text, fid", [
    ("The Q1 2026 adjusted gap was ~24 pts.", "D08"),
    ("The Q1 2026 adjusted gap was about 24 pts.", "D08"),
    ("The Q1 2026 adjusted gap was roughly 24 pts.", "D08"),
    ("The Q1 2026 adjusted gap was approximately 24 percentage points.", "D08"),
    ("The Q1 2026 adjusted gap was around 24 points.", "D08"),
    ("The Q1 2026 adjusted gap was ≈24 pts.", "D08"),
    ("The Q1 2026 distortion was about 510 bps.", "D18"),
    ("The Q1 2026 distortion was about 5.1 percentage points.", "D18"),
    ("The Q2 2026 tariff drag was about $284 million.", "D15"),
    ("The Q2 2026 build was about 25% of the Q1 2026 build.", "D13"),
    ("The Q2 2026 build was roughly a quarter of the Q1 2026 build.", "D13"),
    ("The CI build was roughly three quarters of the company-wide build.", "D14"),
    ("The CI build was about 75% of the company-wide build.", "D14"),
    ("The implied Q1 2026 swing was about $1.3 billion.", "D21"),
    ("CI margin fell ~-470 bps in Q4 2025.", "D23"),
    ("CI margin fell by about 470 bps in Q4 2025.", "D23"),
])
def test_allowed_equivalents_pass(packet, text, fid):
    assert display_codes(packet, text, fid) == set()


def test_allowed_equivalent_passes_the_whole_validator(packet):
    s = st("A", "fact", "The Q1 2026 adjusted gap was about 24 pts, a directional reading.", fact_ids=["D08"])
    lim = st("L", "limitation", "The gap is directional and mixes price bases.", fact_ids=["D08"])
    assert validate_statements(packet, [s, lim]).codes == set()


def test_marker_must_come_from_the_packet_policy(benchmark):
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    policy["display_policy"]["approximation_markers"] = ["~"]
    strict = build_brief_packet(benchmark, generated_at=T0, policy=policy)
    assert display_codes(strict, "The Q1 2026 adjusted gap was ~24 pts.", "D08") == set()
    assert display_codes(strict, "The Q1 2026 adjusted gap was about 24 pts.", "D08") == {
        "APPROXIMATION_MARKER_MISSING"}


def test_marker_must_sit_next_to_the_figure(packet):
    text = "About the Q1 2026 gap: it was 24 pts."
    assert display_codes(packet, text, "D08") == {"APPROXIMATION_MARKER_MISSING"}


@pytest.mark.parametrize("text, fid, expected", [
    ("Disclosed CI dealer inventory rose $1.5 billion in Q1 2026.", "S23", set()),
    ("Disclosed CI dealer inventory rose $1,500M in Q1 2026.", "S23", set()),
    ("Tariffs cut Q2 2026 CI margin by 340 bps.", "S29", set()),
    ("Q2 2026 CI margin was 23.3%.", "S17", set()),
    ("World retail grew 7% in Q1 2026.", "S21", set()),
    ("Q2 2026 CI margin was 23.30%.", "S17", {"FALSE_PRECISION"}),
    ("World retail grew 7.0% in Q1 2026.", "S21", {"FALSE_PRECISION"}),
    ("Reported CI sales growth was 34.8% in Q2 2026.", "D03", set()),
    ("Q2 2026 CI segment profit rose $703M.", "D11", set()),
])
def test_exact_values_behave_as_before(packet, text, fid, expected):
    assert display_codes(packet, text, fid) == expected


def test_uncited_canonical_rendering_is_still_uncited(packet):
    assert display_codes(packet, "The Q1 2026 adjusted gap was ~24 pts.", "S21") == {"UNCITED_FACT_VALUE"}
