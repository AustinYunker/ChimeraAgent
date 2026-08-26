# CHIMERA-agent: design strategy and implementation plan

## Context

We are building a competition entry for the **CHIMERA-agent** challenge (MICCAI 2026, Radboudumc + Karolinska), targeting **all three tasks** with a **hybrid architecture**: supervised models produce the decision and the deterministically-scored reasoning fields, while a real MCP-driven LLM agent gathers evidence and writes the free-text rationale.

The deadline structure is unforgiving and drives everything below:

| Date | Event |
|---|---|
| ~~Aug 10~~ **Sep 1, 2026** | Validation opens — **5 submissions total**, best counts *(pushed back by the organizers)* |
| **Sep 10, 2026** | Test set — **one single submission**, no retries |
| Dec 18, 2026 | Debug phase closes — unmetered, 3/day, a subset of the released dev data |
| Sep 27–Oct 1 | MICCAI; 6-page LNCS paper + public repo required for ranking |

Roughly five weeks, with a one-shot final submission. That inverts the usual priority order: **contract conformance and infrastructure must be proven before modelling starts**, because a container that fails to run on Sep 10 scores zero regardless of model quality.

### Why hybrid rather than agent-first

Reading `evaluation/evaluate.py`, the Task 1/2 case score is a weighted sum of six components, of which **0.75 is deterministic and directly predictable** (confidence 0.20, variable_weights 0.25, important/decisive F1 0.15, tool efficiency 0.15, section grounding 0.15) and only 0.10 is the LLM rationale judge. Those deterministic components are annotations from a urologist form — they are supervised targets with learnable structure, not something an LLM should be guessing at. Meanwhile Task 3 ranks on **C-index alone**, so its reasoning trace is worth exactly zero leaderboard points.

An LLM asked to emit all of this in one shot leaves measurable points on the table. Separating *decide* from *narrate* is both higher-scoring and easier to validate.

### Design rule on `reveal_sequence` (important)

`reveal_sequence` is self-reported and scored, and the evaluator never inspects real MCP traffic. We will **not** exploit that. The reveal policy is chosen *first*, the agent then actually retrieves exactly those sections via MCP, and we report what we retrieved. This keeps the declared trace truthful, keeps the paper defensible, and still captures the points — the optimisation is over *which evidence to acquire*, which is a legitimate agent design question, not over what to claim afterwards.

---

## Environment constraints discovered

| Constraint | Consequence |
|---|---|
| **No Docker** (only singularity-ce 4.1.2; `/etc/subuid` empty → rootless podman/buildah unavailable) | Build the submission image in **GitHub Actions**, download the `docker save` tarball artifact |
| **2× Tesla V100-SXM2-32GB, compute capability 7.0** | vLLM 0.25 (baseline pin) will not run on Volta. Dev locally against an **OpenAI-compatible endpoint** (llama.cpp / Ollama) using the baseline's existing `provider: openai` path |
| GC target GPU is T4 (16 GB, sm_75) or **A10G (24 GB, sm_86)** | Choose **A10G** — sm_86 is safely inside vLLM support and 24 GB removes quantisation pressure |
| `/` has 1.6 TB local disk; NFS home is 98% full | Keep model weights, caches, and builds under local scratch, not NFS |
| HuggingFace reachable; 48 cores; 125 GB RAM | Fine for training and for running the evaluator natively |

We cannot run the official evaluator *container*, but we can run `evaluate.py` **natively** in a conda env with Ollama serving the judge model on a V100. That preserves full-fidelity scoring locally.

---

## Architecture

```
/input  (GC flat sockets + inputs.json, one case per invocation)
   │
   ├─► contract adapter            — socket slugs → internal case record
   │
   ├─► MCP evidence server         — official interface; six section tools + guidelines RAG
   │        ▲
   │        │ executes exactly the chosen reveal set
   │   reveal policy               — picks the section subset to acquire
   │
   ├─► evidence extraction         — structured vars + parsed report fields + frozen embeddings
   │
   ├─► decision head (per task)    — T1 binary · T2 four-class · T3 survival risk
   ├─► reasoning head (T1/T2)      — confidence + per-variable weight distributions
   │
   ├─► decision-theoretic selector — argmax expected component score (see below)
   │
   ├─► LLM writer                  — free_text, grounded in retrieved evidence only
   │
   └─► /output  two flat JSON files per case, exact challenge shapes
```

### The decision-theoretic selector

This is the piece with no analogue in the baseline and the clearest source of edge. Given predictive *distributions* over the urologist's annotations, the scored components are closed-form functions of our output, so we can maximise expected score directly rather than emitting a point estimate:

- **Confidence** (0.20): ordinal distance on a 3-point scale → pick the argmax of expected score over three candidates.
- **Reveal set** (0.15 tool + 0.15 grounding): only **2⁶ = 64 subsets**. Enumerate all of them, compute expected `|R∩R_gt|/|R|` plus the grounding fraction implied by our weight vector, take the argmax. Exhaustive and exact.
- **Variable weights** (0.25 ordinal error + 0.15 set-F1): these two pull in opposite directions — ordinal error rewards hedging toward the middle, set-F1 rewards committing to `important`/`decisive`. Optimise them jointly against the combined weighted objective.

Note the built-in tension the selector has to resolve: an empty reveal set scores a perfect 1.0 on tool efficiency but leaves every non-`psa`/`age` variable ungrounded. The optimum is interior and case-dependent.

### Per-task modelling

- **Task 1** (n=91 labeled of 195): binary. Features are the structured panel (`psa`, `psad`, `psav`, `vol`, `age`, `pirads`, `dre`, `ct`, `cspca`, `bx`, `pmhx`) plus the 1024-d MRI embedding, dimensionally reduced. `cspca` is already a deep-learning score and is likely the single strongest feature. Small *n* ⇒ regularised linear / shallow GBM, nested CV, no deep nets.
- **Task 2** (n=72 labeled of 153): four-class. **Guideline-first.** EAU/NCCN risk stratification from ISUP, PSA and cT is a deterministic decision tree, and the four management options follow it fairly mechanically once comorbidity and life expectancy gate `watchful_waiting`. Encode the guideline explicitly, then learn a small correction model on the residuals. With 72 examples across 4 imbalanced classes, a purely learned model will not beat an encoded guideline.
- **Task 3** (n=75, all labeled): survival regression. Features are the surgical pathology fields (pT stage, margins, SVI, EPE, ISUP, nodal status) plus preoperative PSA and the prostatectomy WSI embeddings. **Only the ordering of predicted months matters** for C-index — `event` and `free_text` need to be schema-valid but contribute nothing to rank. Budget effort accordingly.

Labels "grow incrementally" per the challenge site, so the training pipeline must be re-runnable as new labels land.

---

## Repository layout

```
chimera/
  contract/        socket adapter, output writers, schema validation
  mcp/             MCP evidence server (six section tools + guidelines)
  evidence/        report parsing, structured feature assembly, embedding loaders
  policy/          reveal policy + decision-theoretic selector
  models/          task1/, task2/, task3/ — training + inference
  writer/           LLM free_text generation
  scoring/         fast in-process replica of the official scorer
  cli/             local batch runner; inference.py GC entrypoint
tests/             contract conformance + scorer-parity tests
.github/workflows/ build-image.yml → docker save artifact
```

### What to reuse rather than rebuild

From `DIAGNijmegen/chimera-agent-baseline` (clone fresh; do not vendor):

- `inference.py` — the GC flat-socket adapter, including the **50-char slug truncation** handling for the Task 3 clinical slug. Adapt, don't reinvent.
- `src/chimera_agent_baseline/mcp_server.py` + `tools/definitions.py` — the `ToolSpec` pattern for MCP tools.
- `scripts/aggregate_predictions.py` — rebuilds `predictions.json` from per-case runs, which the official evaluator requires as input.
- `resources/guidelines_db/` — pre-built Chroma index over the EAU corpus, ships in the repo.
- `Dockerfile_Baseline` — image structure and the `/opt/ml/model` weights mount point.

From `DIAGNijmegen/CHIMERA-agent`: `evaluation/evaluate.py` is the **only** pipeline permitted for reporting performance. Treat it as read-only ground truth and never reimplement its semantics — our fast scorer must be verified against it, not trusted on its own.

---

## Checkpoints

Each checkpoint has a binary pass condition. Do not proceed past a failing one.

### C0 — Contract conformance ✅ *(target: Aug 8 — met Aug 8)*
Stand up the repo, run the official evaluator natively, and push a **constant predictor** (fixed decision, fixed reasoning) end to end.

> **Pass:** official `evaluate.py` consumes our output files and emits `metrics.json` with a non-null `ranking_score` for all three tasks, running fully offline.

This de-risks the entire I/O contract — the `prostate-biospy-decision` misspelling (since corrected upstream; we now write both spellings), the flat four-key reasoning object versus the baseline's richer Pydantic shape, bare-JSON-value decision files, and `predictions.json` aggregation — before any modelling exists.

### C1 — Scorer parity ✅ *(target: Aug 10 — met Aug 8)*
Implement the fast in-process scorer for CV.

> **Pass:** fast scorer agrees with official `evaluate.py` to within 1e-9 on every sample case in both repos, with the judge disabled; and within judge noise with it enabled.

**Met, judge-disabled, at two levels.** `chimera/scoring/fast.py` is a transcription of the deterministic scorer; `chimera/scoring/records.py` reproduces the evaluator's own record flattening, so both sides consume byte-equal inputs and any disagreement is a real disagreement about the maths rather than about plumbing.

- `tests/test_scorer_parity.py` drives both scorers over the same in-memory records: randomised hostile cohorts (3 tasks × 5 seeds × 60 cases, plus 400 single cases per task), and the degenerate cohorts that break aggregation — every case failing the gate, every prediction missing, every Task 3 case censored.
- `tests/test_run_dir_parity.py` closes the loop through the filesystem: build a run directory with `run_local`, score it with the real `evaluate.py` as a subprocess, and diff against the fast scorer reading the *same directory* back. This is what catches slug, routing, nesting and case-id faults that in-memory parity cannot see.
- `python -m chimera.cli.score_fast --run <dir> --compare` is the same diff as a command, for use after any `scripts/score.sh` run.

The judge-enabled half of the pass condition is deliberately deferred to **C4**, where the LLM writer first produces rationales worth judging. Until then `mean_rationale_score` is `None` on both sides and comparing it would be theatre.

**Two evaluator crashes found and pinned.** `evaluate_case` keeps *raw* prediction values on a schema-gate failure but normalised ones on success, and the aggregator then does arithmetic on them. A Task 3 case with string `months_to_recurrence` and an invalid `event`, or a Task 2 case with a truthy non-string `treatment_recommendation.primary`, raises during aggregation and loses **the whole task**, not the one case. Our validation cannot emit either shape; `tests/test_contract.py` pins both so an upstream fix shows up as a failing test. Worth reporting to the organizers alongside the other open questions.

### C1b — Early plumbing submission *(on Aug 10, validation open)*
Submit the C0 constant predictor as **validation submission 1 of 5**.

Deliberate spend. With only one test submission, proving the container actually executes on GC infrastructure a month early is worth far more than a marginally better model later. It also reveals real runtime and memory headroom.

### C2 — Decision models *(target: Aug 20)*
Task 1/2 classifiers and the Task 3 survival model, under nested CV.

> **Pass:** CV `ranking_score` beats both a majority-class baseline and the reference agent baseline on all three tasks.

### C3 — Reasoning heads + selector ❌ *cancelled Aug 24* — ⚠️ **cancellation void, re-open**

> ⚠️ **The measurements below were taken with the wrong weights and do not support the
> conclusion drawn from them.** Upstream `192c39c` (Aug 24) repriced section grounding
> 0.15 → 0.05 and rationale 0.10 → 0.20, and reweighted the tasks 1:1:1 → 2:2:1. The
> sweep behind this table priced grounding at **0.175** — 3.5× the live value — and the
> Task 2 head it blessed was tuned to hold exactly that component at 1.000. Every number
> in this section is stale, including the "under +0.005 overall" that justified the
> cancellation. See `docs/debug-phase-notes.md`. The scorer has since been corrected;
> the sweep has not been re-run.

> **Original pass condition:** measurable CV lift in `mean_case_score_among_gate_passed`
> from per-case confidence and variable-weight predictors plus a 64-subset reveal
> optimiser.

**Cancelled: the deterministic reasoning side is already at its practical ceiling.**
Measured against v3, gated to the cases whose decision is correct (the only ones that
pay), in per-task `ranking_score` points:

| | oracle, perfect labels | best feature-conditioned, **in-sample** |
|---|---|---|
| Task 1 confidence | +0.0192 | **+0.0006** |
| Task 2 confidence | +0.0094 | **+0.0000** |
| Task 1 weights + reveals | +0.0309 | **+0.0027** |
| Task 2 weights + reveals | +0.0245 | **+0.0017** |

The right-hand column is an *upper bound* — it strata-fits on the evaluation data
itself, with no held-out penalty — and it still totals under +0.005 overall. Out of
fold it is indistinguishable from zero. The oracle column is much larger only because
a per-case oracle can absorb individual annotator idiosyncrasy, and no feature we
extract predicts that. The gap between the two columns is irreducible label noise, not
unexploited structure.

Three structural reasons, each worth carrying into the paper:

- **`cost_aware_tool_score` returns 1.0 for an empty `reveal_sequence`,
  unconditionally** (`evaluate.py:1073`) — it is precision, and precision over an
  empty set is defined as no cost incurred. Declaring nothing therefore *weakly
  dominates* on that component: it always scores the maximum, and a perfect reveal
  set can only tie it. The sole incentive to reveal anything is `section_grounding`.
- **Task 2's reference `reveal_sequence` is empty in all 72 cases**, so the tool
  component there is binary — 1.0 for revealing nothing, 0.0 for revealing anything.
  Combined with grounding, that leaves exactly two candidate regimes, and the search
  over both is exact rather than approximate. 200 random restarts converge on the
  head we already ship. Its `variable_weight_weighted_kappa` of 0.139 is not a defect;
  it is the correct price for `tool = 1.000` and `grounding = 1.000`.
- **Confidence is scored `1 - |distance| / 2`**, a soft ordinal. A well-chosen constant
  is within a thousandth of the best conditioned policy, because being one step wrong
  still earns half credit.

The reveal optimiser is likewise dead: Task 1 already scores tool 0.946 / grounding
0.976 and Task 2 scores 1.000 / 1.000.

**Where the points actually are, in descending order:** Task 1 decisions (71.4%
accurate, 26 of 91 wrong, every univariate AUC near chance), Task 3's C-index (0.737
from an unfitted nomogram, and a fifth of the overall score with *no* reasoning
component at all), and Task 2 decisions (80.6%, 14 of 72 wrong). One additional
correct Task 2 decision is worth roughly as much as the entire Task 2 reasoning
oracle.

> ⚠️ Task 3 was **a third** of the overall score when this was written; `192c39c`
> reweighted the tasks 2:2:1, cutting it to a fifth and raising Tasks 1 and 2 to
> two fifths each. The ordering above survives — it moves *against* Task 3 and
> *for* the two decision tasks — but the rationale component those tasks now carry
> at 0.20 is not priced anywhere in this section.

### C4 — Agent integration *(target: Sep 1)*
MCP server, reveal execution, LLM writer, offline model weights.

> **Pass:** full pipeline runs offline on a held-out split; declared `reveal_sequence` exactly matches observed MCP calls (assert this in a test); `mean_rationale_score` at or above baseline.

### C5 — Container + GC dry run ✅ *(target: Sep 5 — met Aug 24)*
GitHub Actions build, lean image plus separate model tarball.

> **Pass:** image runs with `--network none` on a single flat-socket case within the GC time and memory budget, on sm_86-equivalent hardware.

Met ahead of schedule, and on stronger evidence than the pass condition asks for: the
**debug phase** — a separate submission pool from the 5 validation slots, 3/day,
closing Dec 18 — accepted `v0.2.1` on **all three interfaces** on Aug 24. That is the
real platform rather than a local proxy. The one failure along the way was the Task 1
socket spelling (see `spec.LEGACY_OUTPUT_FILENAMES`), which is exactly what an early
plumbing run exists to catch.

Debug submissions are cheap and unmetered against validation. Every container change
from here goes through debug before it goes anywhere else.

### C6 — Validation spend *(Sep 5–8)*
Submissions 2–5, staged so each answers a distinct question rather than chasing noise.

### C7 — Test submission + paper *(Sep 8–10)*
One shot. Freeze the model, submit, write the LNCS paper against the frozen artefact.

---

## Verification

- **Contract tests** (`tests/`): assert exact output filenames including both spellings of the Task 1 socket, bare-value decision files, the exact four reasoning keys, and enum token casing. These run in CI on every commit.
- **Scorer parity test**: fast scorer vs official `evaluate.py` on repo fixtures, as C1.
- **Reveal honesty test**: assert the emitted `reveal_sequence` is exactly the set of MCP tools actually invoked. This is the guard on our design rule.
- **Offline test**: run the built image under `--network none` and assert no outbound connection attempts.
- **Domain-shift check**: 100 of 250 test cases come from Karolinska with different scanners, staining, and report templates. Hold out a synthetic-perturbation split (reworded reports, missing sections, out-of-range values) and confirm the pipeline degrades gracefully rather than crashing or silently mispredicting.
- **Missing-modality test**: embeddings arrive as `[]` when a modality is absent, and Task 3 fields are frequently `null`. Every model must have a defined path for each missing combination.

---

## Open questions for the organizers

Worth emailing `nadieh.khalili@radboudumc.nl` early — answers affect the architecture:

1. Does a **deterministic MCP orchestrator** satisfy "the official MCP interface must be used for tool access", or is an LLM-driven loop required? This directly gates the hybrid design.
2. ~~Confirm the **`prostate-biospy-decision` slug spelling** on the live platform.~~ **Answered Aug 24** by the debug submission: the platform wants the corrected `prostate-biopsy-decision`. Only the debug phase has been observed, so we write both spellings until validation and test confirm.
3. Is **A10G selectable** per submission, and is there a per-case wall-clock limit? The baseline's `step_timeout` of 900 s per graph step implies a generous budget, but this needs confirming before we size the model.
4. Report the **two aggregation crashes** in `evaluate.py` (C1). Not a blocker for us, but any participant who trips one loses an entire task's score to an exception rather than to a bad prediction, and the fix is a two-line normalisation in the schema-failed branch of `evaluate_case`.

## Risks

- **Single test submission.** Mitigated by C1b, C5, and freezing early.
- **Small labeled sets** (91 / 72 / 75). Nested CV throughout; prefer encoded guidelines and regularised models over anything high-variance. Expect noisy CV and do not chase sub-point differences.
- **Data not yet in hand.** Build C0–C1 entirely against repo fixtures so the critical path does not idle; swap in real data when it lands.
- **Volta/vLLM mismatch.** Local dev never exercises the exact container inference stack, so C5 is the first real test of it. Do not let C5 slip.
