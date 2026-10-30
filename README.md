# FP&A Intelligence Engine

The AI writes the brief. Code does the math and decides whether the writing is accepted.

The model only sees the benchmark's facts and evidence and can only write structured statements that point to them. A deterministic validator checks each statement: a number it writes must match a fact it cites (at the precision it is written to), a reconciliation it cites, or a figure in validated evidence it cites; text in quotation marks must be an exact excerpt of the curated source excerpt stored in the benchmark; and every ID it cites must exist in the input packet. A statement that fails is rejected. That does not mean the briefs are right in judgment. The checks confirm that figures and quotes match the curated evidence, not that the interpretation is the right one, and nothing is compared with the original filings. See [What the validator does not check](#what-the-validator-does-not-check).

- Portfolio page: https://personal-portfolio-sam-felix.vercel.app/projects/fpa-intelligence-engine
- Project status, with what is and is not established: [docs/project_status.md](docs/project_status.md)

## What it produces

This is part of a real brief the engine produced and accepted. It is shown unedited.

**B1 · fact**

> Construction Industries sales rose to \$8,346M in Q2 2026 from \$6,190M a year earlier. The company's sales bridge reconciles, with sales volume contributing \$1,755M and price realization \$309M.

**B5 · interpretive**

> Taken together, a smaller disclosed dealer build alongside faster price-neutral retail growth suggests that Q2 2026 CI growth leaned less on dealer restocking and more on end-user demand than in Q1 2026.

**B6 · limitation**

> The drivers of the \$703M increase in Q2 2026 CI segment profit remain unresolved. The volume and price-realization contributions were not captured, and no manufacturing-cost driver was disclosed, which should not be read as zero. The Q1 2026 and Q4 2025 CI profit bridges also leave residuals for drivers Caterpillar did not itemize.

*Run 1 of 5. Accepted on the first attempt. The full brief has 8 statements; three are shown.*

## How it works

1. **Load and validate.** The loader reads the benchmark's JSON files and checks that their cross-references line up. A validator then checks every input and its lineage back to the evidence. Any error stops the pipeline.
2. **Calculate and reconcile.** Python computes the derived figures with exact decimal arithmetic, using a registry of named formulas. Each reconciliation is then marked reconciled, unresolved or not applicable. The engine never forces a tie-out.
3. **Package.** The facts that pass those input checks go into a fixed, allowlisted and hashed input packet, so every run can be traced to exactly what the model saw.
4. **Write, then check.** The model writes structured statements only, using a pinned prompt (`synthesis_v3`). A deterministic validator accepts or rejects the brief. There is one attempt plus at most one corrective retry. No second AI grades the first, and no embeddings are used for acceptance.

Python owns the arithmetic, display precision, evidence classification, assumption status, assertions, required coverage and acceptance. The model writes the statements; the brief's title is generated from packet metadata.

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

The narrative evidence, assertions, assumptions and policies are hand-curated benchmark data. The engine does not extract anything from source documents. The full rules are in [docs/synthesis_contract.md](docs/synthesis_contract.md).

## What I tested

I ran the engine live five times on one pinned input. All five briefs were accepted on the first attempt, with no retries, no rejections, no provider errors and no number problems flagged. Each brief had 8 to 10 statements.

Pinned configuration: prompt `synthesis_v3`, packet version 2, model `claude-opus-5-5`. Evaluated at commit `f33a569`. One small change, `72d5299`, was added afterward; it changes only how provider errors are recorded.

The live-run records are held outside this repository (the output folder is git-ignored). The figures above come from that evaluation's summary, as recorded in [docs/project_status.md](docs/project_status.md).

## What this doesn't prove

- It is one company and one quarter, with a hand-curated benchmark.
- Five runs show consistency on one input, not general reliability.
- The "28 / 32" bridge figure on the portfolio's Caterpillar page comes from the workbook. The engine independently confirms two bridges (R01 and R02). The three profit bridges (R03 to R05) are left unresolved on purpose, because the data does not support a tie-out.
- Assumption disclosure was verified only by offline tests, because all five live runs used a path with no assumptions.
- Some benchmark entries are known to need correction. They are documented in docs/project_status.md.
- The model chose the severity labels itself, and they varied across runs.

## What the validator does not check

The validator's rules are deterministic pattern and reference checks. A brief can pass them and still be wrong in these ways:

- **Fiscal periods.** Quarter and year labels are skipped when numbers are checked, so a correct figure given the wrong quarter or year passes.
- **Direction and sign.** Numbers are compared without their sign: "fell" for "rose", or "-\$703M" for an increase, passes.
- **Segment or company.** A correct figure attached to the wrong segment passes.
- **Relevance of citations.** Cited IDs must exist and be of an allowed kind; they need not support the sentence.
- **Numbers and arithmetic in words.** "About nine billion dollars", "roughly half" or "volume plus price equals the increase" are not checked.
- **Unlabelled units.** A figure with no unit, or written in "thousand", is matched as if it were in the fact's unit.
- **Paraphrase.** An attributed statement without quotation marks must rest on quoted, validated source evidence, but its wording is not compared with that evidence. A quotation shortened with an ellipsis can drop a clause and still pass.
- **Quotes against the filings.** Quotations are compared with the excerpts stored in the benchmark, not with the source documents.
- **Causal claims.** "Because of", "driven by" or "caused by" are not tested against the evidence. The prompt forbids invented causes; the validator does not enforce it.
- **Judgement words.** Only the evaluative words listed in the benchmark's voice policy are checked, and an opinion written as a fact statement passes if its numbers are cited.
- **Severity.** The severity label on each statement is not validated.

## Run it

Requires Python 3.11 or later. Everything below except the last step runs offline, with no API key.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[test,llm]"
```

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[test,llm]"
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first; it applies to the current window only.

Run the test suite (offline: the model client is replaced by a fake, so no network and no key):

```bash
python -m pytest
```

Windows (PowerShell):

```powershell
python -m pytest
```

The suite has 738 automated test cases (pytest counts each parametrized case separately), and this should report 738 passed. If you install only `".[test]"`, the 18 tests for the live model client are skipped because the Anthropic SDK is not installed, and 720 run.

Run the deterministic pipeline on the Caterpillar benchmark. It prints the loaded facts, validation results, derived figures and each reconciliation's status (add `--json` for machine-readable output):

```bash
python -m fpa_engine.pipeline
```

Build the input packet and run the validator against the bundled example briefs (one valid, fourteen deliberately invalid, each rejected for a named reason):

```bash
python -m fpa_engine.brief_demo
python -m fpa_engine.brief_demo --dump-packet packet.json   # also write the packet to a file
```

### Live synthesis (needs an API key; costs money)

The live step sends the packet to the Anthropic API, so it needs your own Anthropic API key, and each run is billed to that key's account. The key is read from the `ANTHROPIC_API_KEY` environment variable. `ANTHROPIC_MODEL` is optional and defaults to `claude-opus-5-5`. Copy [.env.example](.env.example) to `.env` (git-ignored), fill in your key, and load it into the shell:

```bash
cp .env.example .env        # then edit .env and add your key
set -a; . ./.env; set +a
python -m fpa_engine.synthesizer --out result.json
```

Windows (PowerShell), with the key set for the current session only:

```powershell
$env:ANTHROPIC_API_KEY = "<your key>"
python -m fpa_engine.synthesizer --out result.json
```

To repeat the five-run evaluation (1 to 5 runs are allowed; results go to a timestamped folder under `artifacts/live_synthesis/`, which is git-ignored, or to a folder you name with `--out`):

```bash
python -m fpa_engine.gate_c2 --runs 5
```

Windows (PowerShell):

```powershell
python -m fpa_engine.gate_c2 --runs 5
```

The runner's file name uses an internal development label. It makes live API calls, which are billed to the key's account.

Without a key, both commands stop with an error saying `ANTHROPIC_API_KEY is not set`, before making any request. Never commit a real key.

## Repository layout

```
fpa_engine/                      the engine (Python package)
  loader.py, validator.py        load and validate the benchmark
  calculator.py, reconciliation.py
                                 derived figures and reconciliation status
  brief.py                       the hashed input packet
  synthesizer.py, llm.py         the model call (isolated client)
  brief_validator.py             deterministic acceptance of a brief
  pipeline.py, brief_demo.py     offline command-line entry points
  gate_c2.py                     live-evaluation runner (several runs of the pinned packet)
  benchmark/cat_q2_2026/         the curated Caterpillar Q2 2026 benchmark (JSON)
    briefs/                      example briefs used by the validator demo and tests
tests/                           offline test suite (pytest)
docs/                            project status and synthesis contract
pyproject.toml                   package metadata and dependencies
.env.example                     template for the live-run settings
```

## What I'd do next

Extract evidence from raw filings with its own validator, correct the known benchmark entries under a new evaluation, and add automated testing on every change.
