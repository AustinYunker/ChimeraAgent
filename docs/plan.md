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
| **Cannot build the image here.** Rootless podman is installed and starts, but `/etc/subuid` has no entry for the build user, so it maps a single UID and fails on `FROM` while unpacking the base layer | Build the submission image in **GitHub Actions**, download the `docker save` tarball artifact. *(Corrected Aug 28: this row read "No Docker (only singularity-ce 4.1.2)"; the failure is a UID-mapping one, and the `Dockerfile` header carries the exact `ApplyLayer` error.)* |
| **2× NVIDIA RTX A6000, 48 GB, compute capability 8.6** | sm_86 is inside vLLM's support range, so the Volta workaround this row used to describe is moot. *(Corrected Aug 28 against `nvidia-smi`; the row previously read "2× Tesla V100-SXM2-32GB, compute capability 7.0" and drove a decision — dev against an OpenAI-compatible endpoint — that we made for other reasons anyway.)* |
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

### C3 — Reasoning heads + selector ❌ *cancelled Aug 24* — ⚠️ **cancellation void** — ✅ *re-run Aug 27: +0.0089 overall*

> ⚠️ **The measurements below were taken with the wrong weights and do not support the
> conclusion drawn from them.** Upstream `192c39c` (Aug 24) repriced section grounding
> 0.15 → 0.05 and rationale 0.10 → 0.20, and reweighted the tasks 1:1:1 → 2:2:1. The
> sweep behind this table priced grounding at **0.175** — 3.5× the live value — and the
> Task 2 head it blessed was tuned to hold exactly that component at 1.000. Every number
> in this section is stale, including the "under +0.005 overall" that justified the
> cancellation. See `docs/debug-phase-notes.md`.

#### Re-run, Aug 27

`guideline_params.json` was last fitted Aug 22 (`e26c2be`); the scorer was repriced
Aug 26 (`291a249`), whose own message says it voids what the old prices bought. The
shipped constants were an argmax against prices that no longer exist. Re-running
`fit_models` unchanged recovers:

| | old (Aug 22 prices) | refit | Δ ranking |
|---|---|---|---|
| Task 1 | 0.6342 | 0.6384 | **+0.0042** |
| Task 2 | 0.6423 | 0.6603 | **+0.0180** |
| overall (2:2:1) | 0.6581 | 0.6669 | **+0.0089** |

Ranking scores at live judge-on prices, from the official evaluator's own per-case
components recombined by `chimera.scoring.reprice`; the omitted `0.20 × rationale`
term is provably identical across the two runs, because the judge is never shown
`variable_weights` or `reveal_sequence` and every other field it sees is unchanged.
Confirmed out of fold: the refit procedure's honest repeated-CV scores are 0.6308 and
0.6561, and correcting the frozen old params for the same in-sample optimism
(+0.0076, +0.0042) reproduces both deltas exactly.

**The correction this section needs.** The claim below that Task 2's
`variable_weight_weighted_kappa` of 0.139 "is not a defect; it is the correct price
for `tool = 1.000` and `grounding = 1.000`" was true only at 0.175 grounding. At the
live 0.05 the trade reverses: the refit takes kappa 0.139 → 0.479 and spends grounding
1.000 → 0.219 to do it, and that is worth +0.018. What survives unchanged is the
*reveal* half — Task 2's reference `reveal_sequence` is empty in all 72 cases, so
declaring nothing is a genuine optimum under either pricing, and the refit still
declares nothing in every stratum. It was the weights that were stale, not the reveals.

**Method, for anything measured from here.** `scripts/score.sh` cannot run the judge
on this host, so `evaluate.py` takes its `rs is None` branch and prices grounding at
0.175. Those are not leaderboard numbers, and this refit demonstrates the failure
mode concretely: it *loses* 0.0102 overall under `score.sh` and *gains* 0.0089 live.
`score.sh` now prints both columns, and `chimera.scoring.reprice` carries the
reasoning; never read the judge-off column as a result.

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

### Item 5 — Task 1 decision accuracy ✅ *closed Aug 28: measured, not deferred*

Task 1 sits at 65/91 = 0.714 accuracy with `no` recall of 11/35, and 24 of its 26
errors are false-`yes`. Closed with **no model change**, because the errors turned
out to be concentrated in a place nothing reachable can separate.

They are not diffuse. Among the 91 high-PI-RADS cases:

| prior biopsy | n | yes | no | errors |
|---|---|---|---|---|
| none | 20 | 20 | 0 | 0 |
| negative | 15 | 13 | 2 | 2 |
| **positive** | **43** | **21** | **22** | **22** |

22 of the 24 false-`yes` sit in that last row — 47% of the leaf holding 92% of the
error, split almost exactly in half. That reframes the C2 finding recorded in
`guidelines.TASK1_LEAVES`, whose docstring has been corrected: prior-biopsy status
is an excellent **router** and a useless **classifier**, and the C2 experiment
tested only the second role. A stratification leaf can assign only a constant, so
"does a constant beat PI-RADS here" was the wrong question; the right one is what
separates *within* the 43. Everything reachable was tried and none of it does —
card features are flat, lesion size and growth language do not separate, and biopsy
ISUP grade (taken from the Version 2 answer key, so coverage is complete) is
non-monotone in P(`no`) and fires at 0.50 precision against a 0.55 break-even,
worth **−0.0210**.

One near miss is recorded because it looks like a result: reading the grade out of
the *text* gives 7 of 8 `no` and appears worth +0.0152. It is a selection artifact.
Task 1 is not served `pathology_report` (0/43), so a text-derived grade covers only
11 of the 43, and the cases whose notes happen to state a grade are the ones one
site documents that way. 100 of the 250 test cases are Karolinska.

**Method note.** This is the second finding in a row where the honest answer was
"the cohort cannot tell us". Both were kept as documentation rather than shipped as
a rule, and both docstrings now record the measurement rather than the intuition.

### Item 7 — Task 3 C-index ✅ *Aug 28: +0.0150 on Task 3, +0.0030 overall*

Task 3 had never been touched: `"fitted": false`, CAPRA-S mapped through a hardcoded
linear function. Two facts closed off most of the obvious surface. Because the
C-index depends only on *ordering* and that map is monotone, `MONTHS_AT_ZERO_RISK`
and `MONTHS_PER_CAPRA_POINT` **cannot move the score at all**. And there is no
missingness headroom: all 75 cases parse all 12 CAPRA-S points.

The headroom was structural instead. CAPRA-S is an integer score taking 12 distinct
values over 75 cases, so it **ties 104 of the 1130 comparable pairs** — 9% of the
metric — and the C-index banks a flat 0.5 on every one. The MRI report's AI-predicted
csPCa probability is parsed on 75/75, is *already retrieved* (so it costs no tool
call and no reveal change), and is only weakly correlated with the nomogram
(Spearman 0.488) because CAPRA-S is pathology-only. It resolves those tied pairs at
0.663.

Shipped as `capra_s + 0.99 * cspca`. The weight is a **bound, not a tuned
coefficient**: since `cspca` ∈ [0, 1] the term is strictly under one CAPRA point, so
it orders cases *within* a band and never across one. That is what makes it safe on
a cohort we have not seen — it needs `cspca` only to beat a coin flip within a band,
not to be calibrated the way Radboudumc's model is, and a report omitting the line
falls back to today's ordering exactly.

| | C-index | Δ | 95% CI (paired bootstrap, 4000×) | P(gain) |
|---|---|---|---|---|
| CAPRA-S | 0.7372 | — | — | — |
| **+ csPCa tie-break** | **0.7522** | **+0.0150** | [−0.0067, +0.0396] | **0.90** |

Official evaluator, `work/run/task3-tiebreak`: task1 0.6384 and task2 0.6603
**unchanged and byte-identical**, task3 0.7372 → **0.7522**, overall 0.6669 →
**0.6699**.

Two things kept honestly on the record. The interval crosses zero — with 19 events
nothing on this cohort is individually significant, and CAPRA-S's own 95% CI is
[0.599, 0.856]. And a fuller `0.5*(capra/12) + 0.5*cspca` blend scored **higher**
(0.7690, P(gain) 0.96) but was **rejected**: half its gain comes from reordering
across CAPRA bands, which is the half that assumes Karolinska's csPCa model is
calibrated like Radboudumc's. We took the calibration-free half only.

### Item 8 — Task 1 rationale register ✅ *Aug 29: +0.0400 on Task 1 rationale, +0.0011 overall*

Item 5 closed Task 1's *decision* as measured-out: 22 of the 24 false-yes errors sit
in one leaf that splits 21 yes / 22 no, and nothing reachable beats that coin flip.
The rationale is a separate 0.20 of the case score, and reading the reference
rationales for that same leaf showed they are frequently not interpretations of the
findings at all. **On the prior-biopsy-positive cases 24% of the references are the
urologist naming what is *missing*** — the earlier ISUP grade, or whether the lesion
has grown — against 4% of the never-biopsied ones. Our text answered a question they
were not asking.

Neither missing fact is reachable from the payload. Release Version 3 removed `bx`
and the grade fields from all 195 Task 1 prompts, and Task 1 is served no pathology
report, so the MRI report's own indication line is the only place either can appear.
`extract_prior_context` reads it there. **This costs nothing**: `radiology_report` is
already the Task 1 policy's single declared reveal, and `McpStore.section` memoises,
so the second read is a cache hit — no tool call, no change to `reveal_sequence`.

Three measurements were taken *before* any prose was written, and two of them
changed it:

* An explicitly stated prior-biopsy polarity agrees with the notes-derived status on
  **28 of 28** released cases that state one, and never fires on a never-biopsied
  case. So it may be asserted.
* **3 of 91** reports do quote the earlier ISUP grade. The first draft said "the
  earlier grade is not stated" unconditionally, which would have been a fabrication
  on every one of them; the grade is parsed and cited instead.
* **0 of 91** reports state an interval comparison — so "no comparison is reported"
  is true on the whole released cohort, and is *still* conditioned on the text,
  because 100 of the 250 test cases come from Karolinska templates we have not seen.

`prior_care` is an OR over several phrasings, so a match establishes that some
earlier episode exists without saying which. It gates the gap sentence and
contributes no claim of its own.

Officially scored, judge **on**, `task3-tiebreak` against `task1-prior`. Only
`free_text` differs between the two runs — decision, confidence, `variable_weights`
and `reveal_sequence` are identical case-for-case, and Tasks 2 and 3 are byte-identical:

| Task 1 component | before | after | Δ |
|---|---|---|---|
| `mean_rationale_score` | 0.7369 | **0.7769** | **+0.0400** |
| `mean_case_score` | 0.5761 | 0.5818 | +0.0057 |
| `ranking_score` | 0.6910 | **0.6939** | **+0.0029** |
| tool / grounding / weight-κ / gate | — | — | **unchanged** |

Overall 0.7188 → **0.7199**. The case-score arithmetic checks out exactly:
0.20 × 0.0400 × (65 graded / 91) = 0.0057.

The per-case split is the reason to believe it. The **40 graded cases where the gap
sentence does not fire scored identically** — 0 up, 0 down — which is also a clean
determinism check on the local judge; all of the movement is on the 25 that fire,
at +0.1040 mean. By stratum:

| history the report states | n | mean Δ | up / down / same |
|---|---|---|---|
| **stated positive biopsy** | **13** | **+0.2231** | **9 / 0 / 4** |
| stated negative biopsy | 4 | +0.0500 | 2 / 1 / 1 |
| earlier PI-RADS only | 5 | −0.0400 | 0 / 2 / 3 |
| prior-care phrase only | 3 | −0.1000 | 0 / 1 / 2 |

The gain lands exactly on the stratum the reference analysis predicted, and nothing
regressed there. **Restricting the sentence to the top two rows was considered and
rejected**: it is a cut on n=8 against a judge that is a conservative proxy
(Pearson +0.818 on ordering but 0.118 low on average), which is the same
selecting-on-noise the Task 3 tie-break was scoped to avoid, and it is arguable in
the opposite direction anyway — "no comparison with prior imaging" is most apt where
there *is* a prior imaging study. Left as measured; a candidate A/B, not a fix.

None of the four regressions is the judge objecting to the gap sentence. All four
ask for something else: the PSA trend (`psa_trend`, which we do not reveal), the
age, or a closer match to the reference's shape. On one the judge notes that the
*reference* is the inaccurate one.

### Item 9 — the looser reader on the section we already declare ✅ *Aug 29: +0.0292 on Task 1 rationale, +0.0008 overall*

Route B was going to reveal `previous_notes` to state prior-biopsy status on more
Task 1 cases. Measuring what it would actually add killed it, and pointed at a
free alternative that does nearly all of the same work.

Item 8 filled `PriorContext.biopsy_result` from this module's own strict patterns,
which want the biopsy and its outcome named in one clause. That is rare — 28 of the
91 labelled cases. `chimera.evidence.notes.classify_prior_biopsy` accepts the many
other ways a report says the same thing (a quoted Gleason or ISUP grade, an
established diagnosis, a completed prostatectomy). Reading it over the **same**
`radiology_report` reaches 63. Adding `previous_notes` on top reaches 67.

| prior-biopsy status stated, of 91 | reader |
|---|---|
| 28 | Item 8's strict patterns, `radiology_report` |
| **63** | `classify_prior_biopsy`, `radiology_report` — *no reveal change* |
| 67 | + `previous_notes` — **this is all Route B buys: 4 cases** |

So Route B's marginal contribution is 4 cases against its −0.0045 tool-precision
cost, worth roughly +0.0005 at Item 8's measured rate: a **net loss near −0.0040,
argued from 4 cases.** Rejected on measurement rather than run; the judged cycle was
spent on the free variant instead.

**The swap is justified against the Version 2 answer key, not against intuition.**
V2 still carried `bx`, so the classifier can be scored over all 91 labelled cases:

| reader | accuracy | wrong polarity | abstains |
|---|---|---|---|
| strict patterns (Item 8) | 28/91 | 0 | 63 |
| `classify_prior_biopsy`, report alone | 87/91 | **0** | 2 |
| + `previous_notes` | 91/91 | 0 | 0 |

Report-alone never states the wrong polarity; all four misses under-call. Restricted
to `positive`/`negative` — the only two answers `extract_prior_context` accepts from
it — it is **63 for 63**. `none` is refused: two of the four misses under-call *to*
`none`, which unlike an abstention is an assertion that a man who was biopsied never
was, and `bool("none")` is true, so it would also arm the history-gap clause on
never-biopsied men who have no prior imaging to compare against. Both refusals are
pinned by tests.

Official evaluator, judge on, `task1-prior` → `task1-notes`, `free_text` the only key
that differs (81 of 195 Task 1 cases; Tasks 2 and 3 byte-identical):

| Task 1 | before | after | Δ |
|---|---|---|---|
| `mean_rationale_score` | 0.776923 | **0.806154** | **+0.029231** |
| `mean_case_score` | 0.581814 | 0.585989 | +0.004176 |
| `ranking_score` | 0.693892 | **0.695980** | **+0.002088** |
| `mean_tool_score` | 0.984615 | 0.984615 | — |
| `mean_section_grounding_score` | 0.674725 | 0.674725 | — |
| `variable_weight_weighted_kappa` | 0.604512 | 0.604512 | — |

Overall 0.719905 → **0.720740**. Per case: 8 up, 4 down, 53 unchanged; among the 12
that moved, mean +0.1583, worst −0.100, best +0.500.

**A repetitiveness concern was raised and does not survive the diagnostic.** The gap
clause now fires on 147 of 195 Task 1 cases and 102 of them close with the identical
joint sentence. But all 195 rationales are distinct, and the judge scores each case
independently against its own reference — there is no mechanism by which corpus-level
sameness is penalised. The live question is aptness on those 102, not sameness across
them.

### Item 10 — the rationale tracks its source, not just its polarity ✅ *Aug 30: correctness, not score*

Item 9 handed the prior-biopsy polarity to `classify_prior_biopsy`, validated at 63/63 on
stated polarities against the Version 2 answer key. That validation was of the *polarity*.
It was not a validation of the **claim shape**, and the two come apart: five of the
classifier's six positive patterns never mention a biopsy at all — "known prostate
cancer", "prior prostate cancer diagnosis", "status post prostatectomy" — and rendering
every one of them as "a previous biopsy positive for cancer" asserts a procedure the
section does not describe.

The Aug 30 debug run caught it, and drew the line precisely:

| debug case | what `radiology_report` says | judge |
|---|---|---|
| `0cdfb9410718` | "a previously documented positive biopsy bucket" | accepted |
| `2e0346bce3b3` | "alongside a previously positive biopsy" | accepted |
| `0020cfca66c8` | "Re-evaluation of prior prostate cancer diagnosis" | *"introduces clinical variables not present in the input data … 'a previous biopsy positive for cancer'"* |

`0020cfca66c8` scored a clean **1.0** on Aug 26 — before Items 8 and 9 — with an explicit
*"does not contradict the clinical data or hallucinate facts (Step 3)"*, and **0.7** on
Aug 30. It is the only Task 1 debug case Items 8–9 changed, and the only one that fell.

`PriorContext.states_biopsy` now records whether the report names a biopsy (or quotes a
grade, which implies one), and the rationale says "a previously diagnosed prostate cancer"
where it does not. The gap clause inherits the distinction: having declined to call the
history a biopsy, it no longer names "the earlier biopsy's ISUP grade" as the missing
fact. Nothing reaches the decision — 27 of 195 Task 1 cases change `free_text` and no
other key; Tasks 2 and 3 are byte-identical.

**The local A/B does not measure this and is not what justifies it.** Only 3 of the 27
changed cases are labelled and gate-passed: 2 up (+0.2, +0.1), 1 flat, for
`mean_rationale_score` 0.806154 → 0.810769 and overall 0.720740 → **0.720872**. In the
same run 2 *unchanged* cases moved on judge nondeterminism alone (0.7→0.6, 0.9→1.0), so
the noise floor is the same size as the signal. The justification is that we were
asserting a procedure the source does not state; the number is not evidence either way.

**Debug-confirmed Aug 30 (v0.4.1): score-neutral, and correct anyway.** Re-submitted with
Item 10 as the only variable. `0020cfca66c8` scored `rationale_score` 0.7 before and 0.7
after, and the judge repeated *both* complaints — the ISUP grading and "mentions a
previous biopsy positive for cancer" — against a text that now contains neither string.
The change stands on the reason it was made; the expectation that it would recover the
case's Aug 26 score of 1.0 was wrong, and that 0.3 remains unexplained by anything in our
text. Full read in `docs/debug-phase-notes.md`.

**The ISUP hallucination flag is not ours and was not fixed here.** Three of three
gate-passed Task 1 cases were docked for "hallucinating" an `ISUP grade group 1 (Gleason
3+3)` that appears in neither our output nor the case. Item 8's gap clause, which names
the ISUP grade in order to say it is absent, was the obvious suspect and is not the cause:
**the Aug 26 run raised the identical complaint on two of the same cases when our Task 1
text contained no "ISUP" at all.** The string comes out of the reference rationale and the
judge attributes it to us — the attribution defect already recorded in
`docs/debug-phase-notes.md`. Both cases carrying the clause scored *higher* with it than
without (0.4→0.5, 0.3→0.4), so it stays. Withdrawing it was drafted and reverted.

The v0.4.1 run turned that archaeology into an experiment. `0020cfca66c8`'s text now
contains no "ISUP" at all, and the judge still charges it with `ISUP grade group 1
(Gleason 3+3)` at an unchanged score. The string is traceable: `0020cfca66c8` and `T2-001`
are the **same patient** — age 67, PSA 4.7, PSAD 0.14, PI-RADS 2, `cspca` 0.59620297 to
eight decimals — and `T2-001`'s card carries `bx_isup 1, bx_gl_prim 3, bx_gl_sec 3`, the
only occurrence of that grade in the whole 12-case batch. One case's record is being
attributed to another case's output within a single submission.

### C4 — Agent integration *(target: Sep 1)*
MCP server, reveal execution, LLM writer, offline model weights.

`docs/organizer-email.md` was **sent on Aug 29**. Its Q2 — whether a deterministic
orchestrator over the MCP tools counts as an agent for this challenge — gates the
only open question left in this milestone: whether the "LLM writer" is required at
all, or whether it can be formally retired. Nothing here should be built until the
answer arrives; if none does before validation opens on Sep 1, ship the
deterministic pipeline, which passes every stated condition below.

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

### C6 — Validation spend *(Sep 1–8)*
Staged in **`docs/validation-staging.md`** so each slot answers a distinct question
rather than chasing noise. *(This header read Sep 5–8, which dated from when
validation opened Aug 10.)*

The governing rule: the debug phase is unmetered and returns the platform judge's own
per-case scores, so **anything debug can answer may not consume a validation slot**.
Debug cases are drawn from our training data, which leaves validation exactly one
question — generalisation — in four forms: the text parsers on unseen prose, the
reference-derived components against annotation we did not fit to, the decision
model's gate-pass rate off-training, and our absolute score.

> **Pre-flight, before Sep 1:** nothing since `v0.2.1` (Aug 24) has been through a
> container, and Items 8–9 are source-only. Build from HEAD, debug-submit, and
> **re-run the judge calibration** — the +0.818 correlation licensing every local A/B
> was measured on rationale text that Items 8 and 9 have since changed on 147 of 195
> Task 1 cases.

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
