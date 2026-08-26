# Notes from the debug-phase `metrics.json` files

Two files, both supplied by the platform: our `v0.2.1` submission (overall **0.8193**)
and the current debug leaderboard leader (**0.9548**).

**These are notes only.** Nothing here has been applied to the predictors, the fitted
parameters, or the container. Where a note implies work, it is recorded as something
to *re-check in our own pipeline*, not as a configuration to copy.

## 0. The debug set is four cases per task

Twelve cases total. Every aggregate in both files is noise-dominated — both entries
scored C-index **1.0** on Task 3, which four cases will do. Do not read the 0.8193 /
0.9548 gap as a measurement of anything.

One useful consequence: the case IDs (`T2-001/017/021/044`, `T3-001/002/006/009`) are
**in our local training data**. The debug set is a subset of the released dev data, so
these cases can be reproduced offline exactly.

## 1. The scoring changed on Aug 24, and our run was scored under the new version

This is the finding that matters most, and it came out of *our* file, not theirs.

**Confirmed upstream.** `DIAGNijmegen/CHIMERA-agent` commit
[`192c39c`](https://github.com/DIAGNijmegen/CHIMERA-agent/commit/192c39c) — *"weighing
rationale higher"*, **Mon Aug 24 09:40 UTC 2026** — makes exactly the change inferred
below, plus a second one the metrics files alone did not reveal. Our `refs/` clone was
pinned at `b0ae4eb` (Aug 16) and predates it. The forensic reconstruction is kept here
because it is what found the change, and because it independently verifies that the
deployed evaluator matches the public commit.

`refs/challenge/evaluation/evaluate.py:1558` weights the six reasoning components
`0.20 / 0.25 / 0.15 / 0.15 / 0.15 / 0.10` (confidence, var_weight, factor_f1, tool,
section_grounding, rationale). Fitting weights to the reported per-case component
values and `case_score`:

| weight vector | fits `metrics_ours.json` | fits `metrics_top_score.json` |
|---|---|---|
| **A** — `.20 .25 .15 .15 .15 .10` (our `refs/` clone) | 0 / 7 (max resid **0.060**) | **8 / 8 exact** |
| **B** — `.20 .25 .15 .15 .05 .20` | **7 / 7 exact** | 0 / 8 (max resid **0.088**) |

Both sum to 1.0; both fit their file to 1e-9. This is identification, not curve-fitting
— six round numbers reproducing seven independent case scores exactly.

The two submissions were therefore scored by **different evaluator versions**. B is
current; the top entry's headline 0.9548 was computed under A and **is not comparable
to our 0.8193**.

| component | before | after | |
|---|---|---|---|
| `section_grounding` | 0.15 | **0.05** | **÷3** |
| `rationale` (LLM judge) | 0.10 | **0.20** | **×2** |
| confidence / var_weight / factor_f1 / tool | unchanged | unchanged | |

Grounding is now the *smallest* component and the judge is tied for second-largest.

### 1b. The task weights also changed: 1:1:1 → 2:2:1

The same commit replaced the equal-weighted mean of task ranking scores with a weighted
one — Task 1 and Task 2 weight 2, Task 3 weight 1. This does not show up in any per-case
number, which is why the component fit above did not catch it. It is visible in the
aggregate:

| | equal thirds | 2:2:1 | reported |
|---|---|---|---|
| ours | 0.8495 | **0.8193** | 0.8193 |
| theirs | 0.9457 | 0.9548 | **0.9548** |

Each file matches exactly one aggregation, and the split is the same one as the
component weights — further confirmation the two runs straddle the commit.

**Task 3 fell from 33% of the overall score to 20%; Tasks 1 and 2 rose from 33% to 40%
each.** That is a roadmap change, not a tuning change. I had previously argued Task 3's
C-index was a priority partly because it was "a full third of the overall score with no
reasoning component at all". It is now a fifth, and it is already our strongest task
(C-index 0.737). Task 1 — 71.4% decision accuracy, and now 40% of the score — is the
single most valuable target we have.

Useful conversion for everything below: for Task 1 or Task 2, a case-score component of
weight *w* is worth **0.2 × *w*** of the overall score (case score is half the ranking
metric, the task is 0.4 of the overall). So rationale is worth 0.04 per task, 0.08
across both; grounding is worth 0.01 per task.

### Consequence: the C3 cancellation rests on the wrong weights

`docs/plan.md`'s C3 section cancelled the reasoning-side work on the measurement that
our shipped Task 2 head is the global optimum for a constant policy. That optimisation
used the **judge-disabled** weights (grounding **0.175**). The live weight is **0.05**
— 3.5× smaller. The head was tuned to protect a component that is now nearly free to
give up, so the optimum it converged on is unlikely to still be one. The C3 conclusion
should be treated as void pending a re-run under B, and that re-run is our own
computation, not an import.

## 2. The judge is enabled, and it only ever sees the clinical-data socket

`mean_rationale_score` is non-null throughout both files, so the judge runs on the
platform. This also answers question 5 of `docs/organizer-email.md`, which can be cut.

More usefully, `evaluate.py:1354` builds the judge's evidence context as
`"clinical_data": pred.get("clinical_data", {})` — the clinical-data **socket**, backfilled
from ground truth at `evaluate.py:578`. It never includes `structured-prompt.json`.

That fully explains our rationale scores. On T2-017 the judge accepted every value we
cited that appears in the clinical-data blob and flagged exactly the ones that do not:

| cited value | in `clinical_data`? | judge verdict |
|---|---|---|
| `PSA 9.0 ng/mL` | yes | accepted |
| `PI-RADS v2.1 score 4` | yes | accepted |
| `PSA density 0.16` | yes | accepted |
| `cT1c` | yes | accepted |
| `age 68` | **no** (structured prompt only) | *"not explicitly stated"* |
| `ISUP 2` | contradicted (report says grade group 4) | *"contradicted by the clinical data"* |

`age` and `bx_isup` are structured-prompt fields. We quote them faithfully; the judge
cannot see them, so they read as invention.

### This reverses what I flagged earlier about Task 3

Task 3 rationale is 0.175, with all four cases accused of fabrication — *"cites PSA 86
ng/mL and age 73, which are not present in the Input data"*. I previously reported this
to you as a correctness problem on our side. It is not. Checking
`work/train/cases/task3/*/structured-prompt.json`:

| case | judge says fabricated | actual input |
|---|---|---|
| T3-001 | PSA 86, age 73 | `psa=86`, `age=73` |
| T3-002 | PSA 0.2, age 54 | `psa=0.2`, `age=54` |
| T3-006 | PSA 11, age 69 | `psa=11`, `age=69` |
| T3-009 | PSA 8.7, age 69 | `psa=8.7`, `age=69` |

Eight for eight, exact. Our Task 3 free_text is accurate; the judge is grading it
against a context that excludes the only file those values live in. Task 3's ground
truth carries no structured prompt at all, so the backfill cannot help either.

Two further Task 3 judge defects, both visible in the reason strings:

- It calls the ground-truth label "the Input" — *"The Input specifies a `reference_event`
  of 1 and `reference_months_to_recurrence` of 1.8"* — and then penalises us for
  predicting something different. That is the task, not a rationale defect.
  `evaluate.py:1315` does put the reference outcome in the judge's input.
- None of it reaches the leaderboard: Task 3 ranks on C-index alone.

## 3. The judge explicitly penalises our boilerplate

Verbatim, across three of four Task 2 cases:

> the rationale *"Weighted most heavily: psa, age"* is unsupported and the statement
> *"Decided from the structured patient record without section retrieval"* is irrelevant
> and potentially misleading

> includes irrelevant information (e.g. *"age 68"*, *"Weighted most heavily: psa, age"*,
> *"Decided from the structured patient record without section retrieval"*)

> overly detailed, includes irrelevant procedural notes

These are fixed strings emitted by our free_text builder. At a rationale weight of 0.20
they are among the most expensive characters in the submission, and removing them costs
nothing else.

## 4. Confidence errors are charged twice

Our `confidence_score` was **0.0** on two Task 1 cases (ground truth `uncertain`, we emit
a constant `clear`). The judge then docked the same cases again — *"fails to align the
confidence level, stating 'clear' when the Expected Output specifies 'uncertain'"* — because
rubric step 5 grades exactly that.

So a confidence miss costs 0.20 deterministically **plus** part of the 0.20 rationale
weight. My C3 analysis priced only the deterministic side, which understated the value
of predicting confidence per case. The top entry has Task 1
`confidence_weighted_kappa` **0.857** against our **0.0**.

## 5. The top entry took the opposite Task 2 corner (observation only)

Both entries got 4/4 Task 2 decisions right, so the Task 2 gap is *purely reasoning with
decisions held constant* — a clean controlled comparison, and the only one the debug set
offers.

| | var_weight | factor_f1 | grounding | tool |
|---|---|---|---|---|
| ours | 0.632 | 0.542 | **1.000** | 1.000 |
| theirs | 0.961 | 0.964 | 0.230 | 1.000 |

They commit to the clinical variables, accept the grounding penalty, and still declare no
reveals — so `cost_aware_tool_score` stays 1.0 for them too. Their `ungrounded_vars` list
(7 of 8 → 0.125) matches the grounding model in the organizer email exactly, which is a
useful independent confirmation that we read that function correctly.

The corner choice is highly weight-sensitive, which is the interesting part:

| | mean Task 2 case score, ours | theirs | gap |
|---|---|---|---|
| under A (grounding 0.15) | 0.8142 | 0.8594 | +0.045 |
| under B (grounding 0.05) | 0.7892 | **0.9264** | **+0.137** |

Under the old weights the two corners were nearly equivalent. Under the current ones,
committing is worth roughly **+0.045 Task 2 ranking points** — an order of magnitude more
than anything the C3 sweep found. Recorded as a reason to re-run our own optimiser under
B, not as a configuration to adopt.

## 6. Follow-ups this creates

Ordered by value and dependency, with rough overall-score estimates. The estimates are
from four cases each and should be treated as direction and order of magnitude, not size.

**First — fix the instrument. Nothing else can be measured until this is right.**

1. **Update `refs/` to `92365b9`** (fetched; working tree not yet moved). Read-only
   reference, no risk.
2. **Point the offline scorer at the new weights** — components `.20/.25/.15/.15/.05`
   *and* task aggregation 2:2:1. Every number in `docs/plan.md` is currently quoted
   against the old scorer.

   ~~Stop `records.py:279-280` backfilling `clinical_data` from ground truth.~~
   **Wrong — do not do this.** That backfill is a faithful transcription of
   `evaluate.py:578-579`, which does exactly the same thing; removing it would break
   parity. The rationale problem was invisible offline for the simpler reason that we
   have never run the judge at all: `scripts/score.sh` defaults to
   `USE_RATIONALE_JUDGE=0` and this host has no Ollama to point it at.

**Then — the changes that pay, in order.**

3. **Rewrite `free_text`** (est. **+0.018**). Strip the procedural boilerplate; cite only
   values that appear in the clinical-data socket. Our rationale is 0.60 / 0.75 against
   the judge's evident ceiling near 0.90, at 0.04 per task. This is the highest-value item
   on the list and the lowest-risk: the strings being removed are ones the judge calls
   *"unsupported"*, *"irrelevant"* and *"potentially misleading"*, and it is hard to argue
   they were earning their place.
4. **Re-run the Task 1 / Task 2 weight and reveal optimisation** (est. **+0.018** on
   Task 2 alone). Grounding at 0.05 instead of 0.175 very likely moves the optimum.
   **Void the C3 cancellation in `docs/plan.md` until this is redone.**
5. **Task 1 decisions.** 71.4% accurate, gating everything, now 40% of the overall score.
   Largest single pool of points left; also the one we have already found hardest to move.
6. **Re-measure per-case confidence** (small). Charged twice — 0.20 deterministic plus
   rubric step 5 — so worth more than C3 priced it. Temper expectations: the earlier sweep
   found almost no *achievable* gain from features even against a large oracle gap. One
   measurement, not a project.

**Separately.**

7. **`docs/organizer-email.md`**: cut question 5 (answered — judge is on), do *not* ask
   about the weights (answered by the commit), and add two: the judge context omitting
   `structured-prompt.json`, and the Task 3 judge treating the reference outcome as an
   input and penalising the prediction for differing from it.

**Note on Task 3.** Its priority drops with the reweighting, but the *reason* to leave it
alone is that it is already at 0.737 with an unfitted nomogram and has no reasoning
component. Its rationale score of 0.175 is free of consequence.

~~**Open gap.** We cannot measure item 3 offline.~~ **Closed (Aug 25).** Ollama is
installed user-local and `scripts/score-judged.sh` runs the official evaluator with the
judge on; see `docs/judge-setup.md`. No debug submissions were spent. Results below.

---

## Status

Items 1 and 2 are done (Aug 25).

- `refs/challenge` fast-forwarded `b0ae4eb` → `92365b9`, so `scripts/score.sh` now applies
  both the new component weights and 2:2:1. Re-scoring `work/run/guideline-v3` gives
  **overall 0.7118**, against the 0.716 previously quoted under equal thirds; the per-task
  rankings are unchanged at 0.698 / 0.713 / 0.737.
- `chimera.scoring.fast` now defaults to the live judge-on pricing of the five
  deterministic components and exposes `CASE_COMPONENT_WEIGHTS_JUDGE_OFF` for parity work.
  Under the new default the selection instrument reprices section grounding at **0.29×**
  what it was, which is the whole point: our Task 2 head was tuned to hold that component
  at 1.000.
- Parity against the official evaluator still holds to 1e-9 on all three tasks
  (`python -m chimera.cli.score_fast --compare`). 173 tests pass.
- `docs/plan.md` carries staleness markers in two places, since every measurement in it
  predates `192c39c`: the C3 cancellation is marked **void** (its sweep priced grounding
  at 3.5× the live value, and the Task 2 head it blessed was tuned to that component),
  and the "a full third of the overall score" claim about Task 3 is corrected to a fifth.

### Item 3 — `free_text` rewritten and measured (Aug 25)

Four judged runs of the **official** evaluator over all 423 released cases, one per
variant, serialised so they did not contend for the resident model. `work/run/judged-*`.

| | baseline | + rewrite | + no DRE | + staging |
|---|---|---|---|---|
| task 1 rationale | 0.5662 | 0.5646 | **0.7369** | 0.7369 |
| task 1 ranking | 0.6747 | 0.6746 | 0.6869 | 0.6869 |
| task 2 rationale | 0.6586 | **0.8362** | 0.8362 | **0.8621** |
| task 2 ranking | 0.6954 | 0.7097 | 0.7097 | 0.7118 |
| task 3 rationale | 0.1214 | 0.2053 | 0.2053 | 0.2053 |
| task 3 ranking | 0.7372 | 0.7372 | 0.7372 | 0.7372 |
| **overall (2:2:1)** | **0.6955** | 0.7011 | 0.7061 | **0.7069** |

Net **+0.0114** against the note's estimate of +0.018. The estimate was optimistic
because it priced only the boilerplate; two of the three columns above are things the
rewrite itself got wrong.

**The rewrite alone bought Task 1 nothing** — 0.5662 → 0.5646, inside noise. Counting the
judge's reason strings over the 65 gate-passed Task 1 cases explains why: `age` complaints
fell 60 → 5 and `generic` 14 → 4, exactly as intended, but *"digital rectal"* appeared in
51 reasons that had not carried it before, and hallucination complaints barely moved
(37 → 33). Verbatim: *"hallucinates a finding by stating 'and an abnormal digital rectal
examination,' which is not present in the Input data"*. Measuring corroboration: the
reports mention a rectal examination in **11.8%** of Task 1 cases — worse than `age` at
22%. The generalised lesson is that **a category is a factual claim too**, not just a
number; `CITABLE` is now framed that way and `tests/test_rationale.py` locks it.

**"Localised prostate cancer" is wrong on the highest-PSA cases.** T2-002 (PSA 187) scored
0.2 and T2-015 (PSA 190) scored 0.4, the judge asking for the staging the reference
rationale calls for. EAU high risk is a stratification *of localised disease* and starts at
20, so the stratum name cannot carry a PSA of 187. Fixed with a clinically anchored
`METASTATIC_CONCERN_PSA = 100.0` — not a swept threshold; 4 of 153 released Task 2 cases
sit above it — worth **+0.0259** Task 2 rationale on its own.

Task 3's rationale improved 0.1214 → 0.2053 and is worth **zero** leaderboard points, as
the note above says. It moved because the same module writes it, not because it was aimed
at.

**The local judge is a conservative proxy for the platform's, not a replica.** The 12
debug cases were scored by the platform under `v0.2.1`, and only `spec.py` and docs have
changed since, so `judged-baseline` carries identical `free_text` — a free paired
calibration. Over the 11 gate-passed cases the two judges correlate strongly on ordering
(Pearson **+0.818**, Spearman **+0.831**, p = 0.002) but ours scores **0.118 lower** on
average (95% CI −0.223 … −0.013). Ordering is the property the A/B above relies on, so
the *direction* holds; the *magnitude* does not transfer, and since scores cap at 1.0 the
offset cannot be a pure shift near the top. **Treat +0.0114 as an upper bound**, with
Task 2 — now at 0.8621 locally — the component most likely to compress. Full table and
caveats in `docs/judge-setup.md`.

**The judge is deterministic on this host.** `judged-rationale` and `judged-nodre` differ
only in Task 1 text (`diff -rq`: 190 biopsy files differ, 0 treatment or recurrence files)
and returned identical Task 2 and Task 3 scores to four decimals over 147 cases. So an A/B
of two rationale variants is a clean read. It is still never a *parity* signal — see the
Reproducibility section of `docs/judge-setup.md`.

**The deterministic side is untouched, verified three ways.** The only files differing
between the old and new runs are reasoning files, and within those the only differing key
is `free_text` (543 decision bodies plus 75 bare-string Task 3 bodies). `score_fast` gives
byte-identical metrics for old and new runs, and `--compare` on a regenerated
`work/run/guideline-v3` still matches the official `metrics.json` to 1e-9. The judge-free
overall is unchanged at **0.7118** on 0.698 / 0.713 / 0.737 — the same three numbers as
before the rewrite. (Not to be confused with the judged Task 2 ranking above, which
happens to land on 0.7118 too.)

Two smaller corrections fell out of the same work:

- `capra_s_points` was split out of `capra_s` so the prose can quote raw points. The
  rescaled score is right for ranking and wrong for prose — a specimen reporting nothing
  but a PSA over 20 rescales to a flat 12 of 12, and stating that would claim a fully
  staged high-risk specimen we never read. All 75 released Task 3 cases take the full-12
  branch; the rescaled branch exists for the Karolinska test cohort and is covered.
- `PriorPredictor` passes no CAPRA-S at all. Its ordering is a cohort constant, so
  quoting a per-case score would be the one thing the rubric does punish — a rationale
  that does not match the prediction.

191 tests pass (173 before).
