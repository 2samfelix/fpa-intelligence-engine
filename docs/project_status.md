# Project status — FP&A Intelligence Engine

Status as of `main` @ `1e6d367` (merge of PR #2, Gate C4). This document describes only what the repository
contains and what was evaluated against it. Labels used throughout:

- **Implemented** — code or data exists on `main`.
- **Tested** — covered by the offline test suite (`pytest`, no network, no API key).
- **Live-evaluated** — exercised against the live model API in a recorded evaluation.
- **Planned** — not implemented; no code on `main`.

## 1. Current project state

| Item | Value |
|---|---|
| Branch / commit | `main` @ `1e6d367` (tree identical to `72d5299`) |
| Scope | One benchmark: Caterpillar Q2 2026, Construction Industries (`fpa_engine/benchmark/cat_q2_2026/`) |
| Benchmark size | 40 sourced facts, 25 derived facts, 7 known disclosure gaps, 13 bridges (5 company-disclosed, 8 not-applicable sets), 24 sources, 17 narratives, 26 assertions, 2 assumptions |
| Brief fixtures | 15 (1 valid, 14 invalid) in `benchmark/cat_q2_2026/briefs/` |
| Offline tests | 735 passing across 20 test files (`pytest`, Python 3.13 venv, no network) |
| Synthesis contract | prompt `synthesis_v3`, packet version 2, packet hash `d726abc65d9d39f8077406bfcc3603596fd0dad36adb5e2421695d9f491409bc` |
| Live-evaluated | Gate C4 synthesis: 5 runs (see section 4; evidence held outside the repo) |
| Continuous integration | None configured in the repository |
| Contract documentation | `docs/synthesis_contract.md` |

## 2. Architecture currently implemented

```
benchmark JSON ──► loader.py (parse, cross-reference checks)
              ──► validator.py (input/lineage validation; blocks the pipeline on errors)
              ──► calculator.py (Decimal arithmetic, named formula registry)
              ──► reconciliation.py (RECONCILED / UNRESOLVED / NOT_APPLICABLE)
              ──► brief.py build_brief_packet (allowlisted, hashed BriefPacket)
              ──► synthesizer.py (prompt synthesis_v3; one attempt + at most one corrective retry)
                    └─ llm.py (isolated Anthropic client; mock client for tests)
              ──► brief_validator.py (deterministic acceptance) ──► ACCEPTED / SYNTHESIS_REJECTED / PROVIDER_ERROR
gate_c2.py — live-evaluation runner (pins prompt version + packet hash; writes allow-listed JSON; secret guard)
```

Division of labour (implemented): Python owns arithmetic, display precision, evidence classification, assumption
status, assertions, required coverage and acceptance. The model writes structured statements only; the title is
generated from packet metadata. No LLM judge and no embeddings are used for acceptance.

Narrative evidence, assertions, assumptions and policies are **hand-curated** benchmark data; nothing is
extracted from source documents by the engine.

## 3. Completed gates

| Gate | Content | Status |
|---|---|---|
| A | Audited Caterpillar benchmark (referenced by Gate B1 commit `50f5e80`) | Data present in repo; the audit itself is not recorded in this repo |
| B1 | Deterministic engine: lineage validation, Decimal calculator, bridge classification (`50f5e80`) | Implemented, tested |
| B2 | Brief packet and deterministic brief contract/validator (`d43f8fc`) | Implemented, tested |
| C | Structured synthesis pipeline, isolated LLM client (`c4350d5`) | Implemented, tested |
| C2 | Live-evaluation runner (`1f5dbf4`); merged as PR #1 (`c37934b`) | Implemented, tested; live C2 run results are not stored in this repo. A live C2 evaluation (5/5 accepted, run 2026-10-13) was observed outside the repository; results are not stored here. |
| C4 | Semantic fidelity contract, steps 0–6, plus provider-error fix (`a7121eb`…`72d5299`); merged as PR #2 (`1e6d367`) | Implemented, tested, live-evaluated (section 4) |

Gate C4 steps (all implemented and tested):
0. Stale workbook-provenance notes updated (hash-neutral).
1. Packet v2: canonical numeric display (`display_step`, `approximate`, `display_wording`).
2. First-class unsourced assumptions (AS01, AS02) with statement-level disclosure.
3. Internal-pipeline-language and packet-ID protection in prose.
4. Role-based required coverage with packet-build feasibility check.
5. Curated assertions; source vs analyst voice; quote, attribution and characterization rules; ordered ellipsis
   fragments.
6. Prompt `synthesis_v3`, final pins, `docs/synthesis_contract.md`, Gate C2-lesson regression audit.
Follow-up `72d5299`: provider HTTP errors record only status, exception class and structured error type.

## 4. Gate C4 live-evaluation evidence

**Provenance:** live-run artifacts are written to `artifacts/live_synthesis/`, which is git-ignored. **No live-run
artifact is stored in this repository.** The figures below come from the externally held `summary.json` for the
C4 evaluation, generated 2026-10-15T20:13Z.

| Measure | Result (from external summary) |
|---|---|
| Contract | prompt `synthesis_v3`, packet hash `d726abc6…`, configured and served model `claude-opus-5-5` for all runs |
| Code under evaluation | `f33a569` (before `72d5299`, which changes only the provider-error path) |
| Runs | 5 |
| First-pass acceptance | 5/5 |
| Final acceptance | 5/5 |
| Retries / hard rejections / provider errors | 0 / 0 / 0 |
| Validator errors on any attempt | none recorded |
| Unsupported numbers / forbidden values / arithmetic in prose | 0 / 0 / 0 |
| Missing mandatory limitations | 0 |
| Model mismatches | 0 |
| Statement counts | 8, 8, 10, 9, 9 |

Per-run records for all five runs and the summary were reviewed outside the repository; none are stored here.
Token usage is not recorded by the runner. An earlier C4 attempt aborted before producing any brief (HTTP 401; see
section 7).

All five accepted briefs used the assumption-free key-interpretation path, so the assumption-disclosure rules in
section 5 are verified by offline tests only, not by a live run.

Five runs are a small sample: they show the contract **can** be met by the configured model, not that it will be
met at a measured rate in production.

## 5. Guarantees the current system actually provides

All guarantees below are implemented and covered by the offline tests. They apply to briefs **accepted by the
validator against the pinned Caterpillar packet**; they are deterministic rule checks, not semantic judgments.

- **Numbers:** every figure in an accepted brief matches a cited packet fact's canonical display (or a registered
  unit conversion, a cited reconciliation residual, or a number in cited explicit evidence); no forbidden value;
  no arithmetic performed in prose; no precision finer than the packet's display step; approximate figures carry
  an approved marker.
- **Classification:** derived facts are not called disclosed; interpretive evidence is not presented as fact or
  explicit claim; unresolved reconciliations are not described as complete; not-applicable sets are not described
  as additive.
- **Assumptions:** a statement relying on an unsourced assumption declares it, is interpretive, uses an approved
  inference marker, and is paired with a limitation citing the assumption.
- **Assertions and voice:** explicit claims cite curated assertions; source-voice wording is attributed;
  attribution requires a source-voice assertion of validated, quoted explicit evidence; quotations are exact
  excerpts (ellipsis fragments in source order); policy-listed characterization words need a supporting assertion
  or assumption.
- **Prose hygiene:** no policy-listed engine vocabulary and no packet IDs in statement text.
- **Coverage:** every required coverage role and every mandatory limitation is present.
- **Contract integrity:** title generated from packet metadata; packet hash and analysis ID must match; statement
  count within bounds; one corrective retry maximum, carrying validator errors and never the rejected output.
- **Live-run hygiene:** the runner refuses mismatched prompt/packet pins and refuses to write any record
  containing the API key or a credential/header marker.

## 6. Known limitations

- **Single benchmark.** All data, policies and pins are specific to Caterpillar Q2 2026 Construction Industries.
- **Curated inputs.** Narratives, assertions, assumptions and policies are hand-written; source excerpts are short
  (39–150 characters) and no full source text is stored in the repository.
- **Not semantic verification.** General paraphrase equivalence is not judged. A meaning-changing paraphrase, or
  an evaluative word absent from the policy list, can pass. Coverage confirms which evidence is cited, not that the
  prose draws the intended conclusion.
- **Pattern-based detection.** Attribution and internal-language detection rely on policy-owned patterns.
- **Evidence gaps.** N13 (Q4 2026 dealer-inventory drawdown) and N14 (FY2026 tariff guidance) have no captured
  quote and cannot be used as evidence.
- **Source status unverified now.** Source URLs record a retrieval date of 2026-10-05; current availability and the
  licensing of third-party call transcripts are not recorded.
- **Small live sample** (section 4); no token-usage measurement; no CI.
- N08's curated quote contains the word "about", which is absent from the Q2 transcript sentence "Dealer inventory increased by $400 million in Construction Industries compared to a decrease in the second quarter of 2025" (manual review, 2026-10-16).
- N14 combines two separate transcript statements and omits the qualifier "excluding the expected IEEPA tariff recoveries of approximately $400 million".
- The N13 and N14 statements do appear in the Q2 transcript (manual review, 2026-10-16) but their quotes are not yet in the benchmark.
- The 340 bps tariff figure appeared in only 2 of 5 live runs.
- The runner's summary gate checks only three criteria.

## 7. Known non-blocking follow-ups from C4

| Item | Description | Effect if fixed |
|---|---|---|
| Statement-kind overlap | A `fact` may cite explicit evidence and a source-voice assertion, so the same content can legally be `fact` or `explicit_claim` (observed in live runs 1, 3 and 4) | Validator and prompt change; new live evaluation required |
| Severity | Statement `severity` is model-selected and unvalidated; the same mandatory limitation varied between `warning` and `critical` across live runs | Policy, validator and prompt change; new packet hash; new live evaluation required |
| Research-process wording | Packet text (CN05; the G07 message; "not_collected" gap prefixes; R04's reason) and one prompt line induce phrases such as "not captured" and "exact-phrase match" in briefs | Packet and prompt change; new packet hash; new live evaluation required |
| Runner naming | Live-evaluation runner and summary fields still carry Gate C2 names (`gate_c2.py`, `gate_c2_passed`) | Cosmetic |
| Package metadata | `pyproject.toml` description still says "Gate B1" | Cosmetic |
| Prompt stricter than validator | Prompt forbids unit conversions that the validator accepts | None required |
| Live-run 401 (resolved) | Aborted live run caused by a 401 error body tripping the secret guard; fixed in 72d5299. | Fixed; no re-evaluation needed |

## 8. Remaining roadmap (all planned; no code on `main`)

| Gate | Purpose | Current state |
|---|---|---|
| D — Narrative extraction | Produce structured narrative evidence and assertions from supplied raw source text, validated deterministically, scored against the curated Caterpillar narratives | Planning audit done outside the repo; prerequisite is a raw-text corpus (none in repo) and a transcript-licensing decision |
| E — Source retrieval | Obtain source documents automatically | Not designed |
| F — Full end-to-end engine | Sources → extraction → packet → synthesis → validated brief | Not designed |
| G — Second-company generalization | Apply the engine to another company/benchmark | Not designed; all current policies are Caterpillar-specific |
| H — Product / UI | User-facing interface | Not designed |
| Final QA | End-to-end quality assurance | Not designed |
| Portfolio integration | Present the engine in the portfolio repository | Not started; the portfolio's Caterpillar case study is a separate artifact |

## 9. What must not be claimed yet

- That the engine verifies meaning or semantic faithfulness in general (only the deterministic rules in section 5).
- That it extracts narrative evidence, retrieves sources, or reads filings or transcripts itself.
- That it works for any company, period or segment other than Caterpillar Q2 2026 Construction Industries.
- That it is production-ready, continuously tested in CI, or measured for cost/token usage.
- That the five-run C4 result is a reliability or acceptance-rate estimate.
- That live-evaluation evidence is stored or reproducible from this repository.
- That every source is currently accessible or licensed for redistribution.
- That any bridge-count figure on the portfolio site is engine output; the engine benchmark contains 13 bridges.
- That N13 or N14 is engine-evidenced: the statements exist in the source transcript, but no quote is captured in the benchmark yet.
