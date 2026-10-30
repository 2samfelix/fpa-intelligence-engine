# Structured synthesis contract (Gate C4)

This document describes how the engine turns a validated analysis into an executive brief, and what is
enforced deterministically before a brief is accepted. It reflects Gate C4 (prompt `synthesis_v3`, packet
version 2).

## Contract identity

The synthesis contract is identified by two pins, which must match together:

| Pin | Value | Where |
| --- | --- | --- |
| Prompt version | `synthesis_v3` | `fpa_engine/synthesizer.py` `PROMPT_VERSION`; `fpa_engine/gate_c2.py` `EXPECTED_PROMPT_VERSION` |
| Packet hash (Caterpillar Q2 2026 CI) | `5eeb00d3342b281452f30eb9cef8f8026d4fa96b1b73a385e9d82c01e574291d` | `fpa_engine/gate_c2.py` `EXPECTED_PACKET_HASH`; `tests/test_brief_policy.py` `CURRENT_HASH`; every brief fixture |

The recorded Gate C4 live runs were made against the earlier packet hash
`d726abc65d9d39f8077406bfcc3603596fd0dad36adb5e2421695d9f491409bc`; correcting N08 and N14 from Caterpillar's own
transcript (PR #5) moved it to the value above.

The packet hash covers every field of the packet except its build timestamp, so any change to facts, display,
assumptions, assertions or policy changes the hash. The live evaluation runner refuses to start unless both pins
match. Earlier prompt versions: `synthesis_v2` (Gate C2 live runs), `synthesis_v2_1` (interim, Gate C4 step 3).

## Flow and division of labour

```
benchmark -> deterministic pipeline -> BriefPacket (allowlisted, hashed)
          -> model writes structured statements -> schema validation -> deterministic brief validation
          -> ACCEPTED, or one corrective retry, or SYNTHESIS_REJECTED
```

Python owns all arithmetic, display precision and approximation status, evidence classification, assumption
status, the curated assertions each narrative supports, required coverage, and the accept/reject decision. The
model's role is structured executive synthesis only: it chooses what to say from the packet, writes it, and cites
it in reference fields. The title is generated from packet metadata, never by the model.

Retry policy: one initial attempt and at most one corrective retry (`MAX_ATTEMPTS = 2`). The retry receives the
same packet plus the deterministic validator errors. It never receives the rejected output. A second invalid
response is a hard rejection. There is no autonomous repair loop.

## What Gate C4 added

Gate C4 closed the five failure modes seen in the Gate C2 live runs.

### 1. Canonical numeric display

Every packet fact carries `display`, `display_step`, `approximate` and `display_wording`.

- Sourced facts keep their disclosed precision and are never approximate.
- Derived facts are rounded no finer than their input-rounding uncertainty. Where the calculator has a linear
  half-width it is used; otherwise the packet builder bounds the formula at every corner of its rounded inputs.
  The display step is the next power of ten at or above that half-width, and the fact is approximate when that
  step is coarser than the normal display. For example, D08 is shown as `~24 pts`, D07 as `~8 pts` and D18 as
  `~510 bps`.
- `brief_policy.json` `display_policy` lists the approved approximation markers (`~`, `≈`, about, approximately,
  roughly, around). It also lists validated per-fact overrides, each with a reason, for example D13 "roughly a
  quarter" and D15 `~$284M`.
- Validator rules:
  - `FALSE_PRECISION`: a figure is finer than `display_step`, for example `24.3 pts`.
  - `APPROXIMATION_MARKER_MISSING`: an approximate figure is written without a marker immediately before it,
    for example `24 pts`.

### 2. First-class assumptions

Unsourced analytical premises are entities in `assumptions.json`, referenced by narratives by ID. In the packet
each has `text`, `sourcing` (`unsourced`), `introduced_by`, `allowed_use` (`interpretive_only`) and
`must_disclose`. Provenance stays in the benchmark. Assumption text that cites the workbook blocks the packet
build instead of being silently trimmed, so an assumption's sourcing status can no longer be lost. Statements
declare `assumption_ids`. Dependencies are computed from references only: a statement depends on every
assumption of every narrative it cites, including those reached through premise chains.

Validator rules:
- `HIDDEN_ASSUMPTION`: the statement depends on an assumption it does not declare.
- `ASSUMPTION_AS_FACT`: an interpretive-only assumption supports a fact or explicit claim.
- `MISSING_ASSUMPTION_DISCLOSURE`: the interpretive statement does not itself use a policy inference marker such
  as "we read" or "consistent with".
- `MISSING_ASSUMPTION_LIMITATION`: a must-disclose assumption has no limitation that cites it.
- `UNKNOWN_ASSUMPTION_ID`.

### 3. Internal-language protection

`brief_policy.json` `prose_policy.internal_language` holds concept-labelled patterns for engine vocabulary:
validator, packet, fixtures, forbidden or rejected values, stale or excluded estimates, retries, schemas,
regression tests and the benchmark dataset. Words that also have a financial meaning are matched only in their
internal contexts. For example, "benchmark index", "fixtures and fittings" and "eliminations are excluded" pass.

Validator rules:
- `INTERNAL_PIPELINE_LANGUAGE`: applies to every statement kind.
- `PACKET_ID_IN_PROSE`: rejects any identifier actually defined in the current packet, matched whole and
  case-sensitively. IDs belong only in reference fields.

### 4. Role-based required coverage

`brief_policy.json` `coverage_policy.roles` lists the analytical spine a brief must carry. For Caterpillar:

| Role | Covered by one statement citing |
| --- | --- |
| headline_evidence | S23 + S25 |
| demand_signal | S21 + S22, or D08 + D07 |
| primary_reconciliation | R01, or R02 |
| material_unresolved_item | R04 |
| key_interpretation | an interpretive statement citing S23 + S25 + S21 + S22, or S23 + S25 + D08 + D07, or N15 |
| mandatory_limitation | the surfacing policy (enforced by `MISSING_MANDATORY_LIMITATION`) |

Each role also restricts which statement kinds may cover it. Coverage is read from reference fields only, and the
IDs of one alternative must sit in a single statement. One statement may cover several roles.
`key_interpretation` does not require the unsourced driver-order assumption (AS01): N15 is optional support,
and AS01's rules apply when it is used. Alternatives marked `availability_dependent` are dropped when one of their
derived facts cannot be computed.

Policy errors block the packet build:
- unknown IDs
- empty alternatives, or a role with none
- duplicate roles
- kinds that cannot legally cite an alternative
- a role with no viable alternative

The build also checks feasibility. The fewest statements that can cover every role is the smallest set of
statement kinds that meets every role's achievable kinds. If that exceeds `statement_bounds.max`, the build fails
with `COVERAGE_INFEASIBLE`.

Validator rule: `MISSING_REQUIRED_COVERAGE` names the missing role and the structured alternatives that would
satisfy it.

### 5. Assertion-based fidelity, quotation and attribution controls

`assertions.json` holds 26 curated assertions, at least one per narrative. Each restates an existing claim or
verbatim quote; none is generated by the model. Voice `analyst` may be stated plainly; voice `source` is the
source's own wording and may appear only when attributed. Source-voice quotes are exact excerpts of the
narrative's verbatim quote, checked at load. Statements cite `assertion_ids`.

Validator rules:
- `EXPLICIT_WITHOUT_ASSERTION`: an explicit claim does not list the assertions it restates.
- `ASSERTION_WITHOUT_EVIDENCE`: an assertion is cited without its narrative.
- `SOURCE_VOICE_NOT_ATTRIBUTED`: source wording is used without attribution.
- `ATTRIBUTION_WITHOUT_EVIDENCE`: attribution language (`voice_policy.attribution_patterns`) appears without a
  source-voice assertion of validated, quoted explicit evidence. This blocks turning evidence gaps, such as the
  Q4 2026 drawdown or the FY2026 tariff guidance, into management claims. A limitation using negation ("was not
  captured") may describe such a gap.
- `QUOTE_NOT_VERBATIM`: quoted text is not an exact excerpt of a cited, validated verbatim quote. Matching
  ignores case and whitespace runs. Ellipsis-separated fragments must keep source order, but need not be
  adjacent.
- `QUOTE_NOT_ATTRIBUTED`: a quotation is not attributed to its source.
- `UNSUPPORTED_CHARACTERIZATION`: a word from `voice_policy.characterization_terms` (typical, strong, robust,
  material, sharp, unusual, healthy, …) appears in a fact, explicit claim or interpretive statement without
  support. Support means a cited analyst-voice assertion uses the word, a declared assumption uses it (for an
  interpretive statement), or a cited source-voice assertion uses it and the statement attributes it.
  Limitations are exempt.

## Full rule set

Every statement must pass all of these. Coverage is an additional layer, never an exception.

- References: `UNKNOWN_FACT_ID`, `UNKNOWN_EVIDENCE_ID`, `UNKNOWN_RECONCILIATION_ID`, `UNKNOWN_PREMISE_ID`,
  `UNKNOWN_FLAG_ID`, `UNKNOWN_COMPARABILITY_NOTE_ID`, `UNKNOWN_ASSUMPTION_ID`, `UNKNOWN_ASSERTION_ID`
- Statement kinds: `FACT_WITHOUT_CITATION`, `INTERPRETIVE_AS_FACT`, `EXPLICIT_WITHOUT_EVIDENCE`,
  `INTERPRETIVE_AS_EXPLICIT`, `MISSING_PREMISES`, `INTERPRETIVE_MARKER_MISSING`, `LIMITATION_WITHOUT_SUBJECT`
- Numbers: `FORBIDDEN_VALUE`, `FALSE_PRECISION`, `APPROXIMATION_MARKER_MISSING`,
  `DERIVED_PRESENTED_AS_DISCLOSED`, `ARITHMETIC_IN_PROSE`, `UNCITED_FACT_VALUE`, `UNSUPPORTED_NUMBER`
- Relationships: `NON_COMPARABLE_PRESENTED_AS_PRECISE`, `UNRESOLVED_PRESENTED_AS_RECONCILED`,
  `NA_RELATIONSHIP_AS_ADDITIVE`
- Assumptions: `HIDDEN_ASSUMPTION`, `ASSUMPTION_AS_FACT`, `MISSING_ASSUMPTION_DISCLOSURE`,
  `MISSING_ASSUMPTION_LIMITATION`
- Prose: `INTERNAL_PIPELINE_LANGUAGE`, `PACKET_ID_IN_PROSE`
- Voice and quotes: `EXPLICIT_WITHOUT_ASSERTION`, `ASSERTION_WITHOUT_EVIDENCE`, `SOURCE_VOICE_NOT_ATTRIBUTED`,
  `ATTRIBUTION_WITHOUT_EVIDENCE`, `QUOTE_NOT_VERBATIM`, `QUOTE_NOT_ATTRIBUTED`, `UNSUPPORTED_CHARACTERIZATION`
- Limitation coverage: `MISSING_FLAG_LIMITATION`, `MISSING_NA_LIMITATION`, `MISSING_RECONCILIATION_LIMITATION`,
  `MISSING_REVIEW_LIMITATION`, `MISSING_DISCLOSURE_GAP`
- Brief level: `EMPTY_BRIEF`, `DUPLICATE_STATEMENT_ID`, `TITLE_NOT_DETERMINISTIC`,
  `STATEMENT_COUNT_OUT_OF_RANGE`, `MISSING_MANDATORY_LIMITATION`, `MISSING_REQUIRED_COVERAGE`,
  `PACKET_MISMATCH`, `PACKET_HASH_MISMATCH`
- Output handling: `SCHEMA_INVALID`, `MISSING_PACKET_HASH`, `MALFORMED_OUTPUT`

## What is not checked

Fidelity is deterministically enforced for structured assertions, quotations, attribution, policy-owned
characterization language, assumptions and references. General paraphrase equivalence is not judged.

- No LLM judge and no embedding similarity are used for acceptance; every decision is a deterministic rule over
  the packet and the brief's text and reference fields.
- A statement's wording is not checked for meaning against the assertion it cites. A paraphrase that changes the
  meaning, or an evaluative word missing from the policy list, can pass.
- Coverage confirms that the right evidence is cited by a statement of the right kind, not that the prose draws
  the intended conclusion from it.
- Attribution and internal-language detection use policy-owned patterns; wording outside those patterns is not
  caught.
- Assertions, assumptions and policies are curated by hand for each benchmark; the engine does not extract
  them from sources.
