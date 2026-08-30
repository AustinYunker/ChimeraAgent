# C6 — how the five validation submissions get spent

Validation opens **Sep 1** and the test set is **Sep 10, one shot**. Five slots,
**best counts**. This document fixes what each slot is for *before* any of them is
spent, because the failure mode here is not submitting a bad model — best-counts
makes that free — it is spending slots on questions that were already answerable
elsewhere, and then having none left when something breaks.

*(The C6 header in `plan.md` said Sep 5–8, which dates from when validation opened
Aug 10. The real window is Sep 1–8, and the extra days matter: the slots have to be
serialised, so the budget is measured in submission turnarounds, not in slots.)*

## The rule that decides everything: what is free, and what is not

| | debug phase | validation |
|---|---|---|
| cost | **unmetered**, 3/day, closes Dec 18 | **5 total, ever** |
| cases | 12 (4/task) | unknown, larger |
| provenance | **a subset of our own training data** | **unseen** |
| returns | per-case component scores **and the platform judge's reason strings** | presumed the same — S1 confirms |

Debug already gave us a paired, free calibration of our local judge against the
platform's (`a4a7f02`, `docs/judge-setup.md`): Pearson **+0.818**, Spearman **+0.831**,
ours **0.118 low**. That is the whole reason the local A/B discipline is trustworthy.

So the rule is: **anything the debug cases can answer is not allowed to consume a
validation slot.** That covers the entire container/contract surface, and — because
debug returns the platform judge's own per-case scores — it covers every question
about the rationale text too.

What debug *cannot* answer is the one thing it is structurally disqualified from: the
debug cases are in our training data, so they say nothing about generalisation. That
leaves exactly four questions for validation, and they are all the same question:

1. Do the **text parsers** hold up on prose we have not seen? 100 of the 250 test
   cases are Karolinska, whose templates we have never read.
2. Do the **reference-derived components** — `tool_score`, `section_grounding`,
   `variable_weight_weighted_kappa` — hold up against annotation we did not fit to?
   Every local value for these is measured against training ground truth.
3. Does the **decision model** hold its gate-pass rate off-training?
4. What is our **absolute** score on a real cohort?

Everything below follows from that list.

## Pre-flight — Aug 30–31, before a single slot is spent

*(Superseded Aug 30: `v0.4.0` is built and debug-submitted, and P0–P3 are all closed
except P2 Q1. The paragraph below records the state P0–P3 were written against.)*

**The last *built* tag is `v0.3.0` (Aug 26), and the last *submitted* image is
`v0.2.1` (Aug 24).** Eleven commits sit on top of `v0.3.0`, and they are not
cosmetic: the entire MCP subsystem — `client.py`, `protocol.py`, `server.py`,
`tools.py`, about 950 lines — postdates it, along with `inference.py`'s rewrite to
reach evidence through the server instead of reading the case directly. **None of it
has ever run inside a container.** Items 8 and 9 are the small part of this.

The MCP layer adds no runtime dependency (it is hand-rolled stdlib precisely so the
SDK's pydantic/anyio/httpx tree stays out of the image; `mcp>=1.0` is a test-only
dep and the image installs `--no-deps`), so the risk is not the build. It is the
stdio subprocess: `McpSession.for_input` spawns `python -m chimera.mcp.server` as a
non-root user against a read-only `/input`, with `/tmp` a separate volume, and that
combination has never been exercised.

- **P0. Make a silent MCP failure fail the build.** *(done — `scripts/smoke_test_image.sh`)*
  `inference.py::open_store` degrades to an in-process `DirectStore` when the
  transport fails. That is the right runtime behaviour — a crashed case is scored
  against a sentinel label and costs the true class its recall, so a lost subprocess
  must cost provenance rather than a case — but it is **invisible in the outputs**.
  `check_outputs` passes either way, because a `DirectStore` prediction is perfectly
  well-formed. Without an assertion, CI would go green on an image that never speaks
  MCP, and we would submit it while asking the organizers whether our MCP
  orchestration is compliant. The smoke test now captures each container's log and
  fails if any case shows `falling back` / `retrying with a direct read`, or omits
  the `mcp server` handshake line.
- **P1. Tag `v0.4.0` and build.** *(done — built, and the `0.4.0` line is in the
  Aug 30 run.)* The version string sat at `0.1.0` through four
  tags; it is now bumped and **logged by `inference.py` on startup**, because the
  platform names a submission by upload rather than by content and a debug result
  that cannot be tied to a commit is not evidence of anything.
- **P2. Debug-submit it.** *(submitted Aug 30 — `metrics_august_30.json`; Q2 and Q3
  answered, Q1 still open. Full read in `docs/debug-phase-notes.md`.)* Three questions,
  in order of what would hurt most:
  1. **Does the MCP subprocess survive the platform's sandbox?** The `mcp server`
     line in the returned logs is the whole answer. If it is absent, every case
     silently ran on `DirectStore` and the architecture question moves from
     "compliant?" to "not implemented".
     **Unanswered.** `metrics.json` carries no logs, and the outputs cannot decide it:
     a `DirectStore` fallback is byte-identical by construction, which is exactly why
     the bit-for-bit agreement below proves nothing here. Ask the organizers for the
     container logs, or fold the handshake into something the outputs can carry.
  2. Does the container still run clean — sockets, exit codes, wall clock?
     **Yes, and better than "clean".** All 12 cases returned, and every deterministic
     component matches the local run on every case: 84 comparisons across `gate`,
     `tool_score`, `section_grounding_score`, `variable_weight_score`,
     `confidence_score`, `important_decisive_factor_score` and `decision_score`, **0
     mismatches**. The local scorer *is* the evaluator on everything but the judge, so
     no slot need ever be spent on a deterministic component.
  3. What does the platform judge score the new rationale?
     Task 1 moved two of three gate-passed cases up and one down. The one that fell had
     been a clean 1.0 on Aug 26, and the judge quoted our own Item 9 phrasing back
     accurately — see Item 10 in `plan.md`.
- **P3. Re-run the judge calibration against P2's returned scores**, paired, exactly
  as `a4a7f02` did. *(done — the offset replicated to four decimals, −0.118 →
  **−0.1182**, across a rewritten rationale. All-task Pearson fell +0.818 → +0.489, but
  entirely on Task 3, which carries zero rationale weight, and on a Task 2 spread of
  {0.9, 1.0} where ordering is noise. **Task 1 ordering is perfectly preserved**
  (Pearson +0.982, n=3), so the local judge keeps its licence to rank Task 1 variants —
  it just sits ~0.167 high in level, which makes Items 8–9's +0.0693 an upper bound
  rather than an estimate. The keep/drop decision this step was staged to force does
  not fire: only ranking was ever relied on, and ranking held.)*

**P3 is the one that is easy to skip and should not be.** The −0.118 offset and
+0.818 correlation were measured on the *pre-Item-8* rationale. Items 8 and 9 changed
the shape of that text on 147 of 195 Task 1 cases, adding a closing sentence that did
not exist when the instrument was calibrated. The local judge's licence to rank
variants was established for text we no longer emit. If P3 shows the correlation has
decayed, then the **+0.0693 that Items 8 and 9 booked is not established** and the
first decision of the whole validation window is whether to keep them — a decision
that would otherwise be made silently by shipping. n is only ~3 Task 1 cases, so this
is a smoke test and not a measurement; treat a collapse as informative and a pass as
merely non-alarming.

## Slot allocation

Two invariants across all five:

- **Serialise.** Never submit a slot before reading the one before it. Two slots in
  flight is two slots spent on one question.
- **One variable per slot**, changed from a scored predecessor. `reveal_sequence`
  honesty is not a variable — it holds in every submitted artefact, per the design
  rule in `plan.md`.

### S1 — Sep 1, opening day. The frozen current best, unmodified.

Submit exactly what P2 put through debug. Its job is not to score well; it is to
produce the diagnostic that aims S2–S4. Three things come back:

*(Aug 30: Item 10 has since changed `free_text` on 27 of 195 Task 1 cases and nothing
else — Tasks 2 and 3 are byte-identical. It is a correctness fix, not a tuning change:
we were asserting a biopsy the source never mentioned. It goes through debug first, and
S1 then freezes on whatever debug last scored, which keeps the invariant intact. The
Task 1 numbers in the table below are pre-Item-10; the post-Item-10 values are
`mean_rationale_score` 0.8108 and overall 0.7209.)*

- **the cohort size** — if validation turns out to be debug-sized, every delta below
  is noise and this plan collapses to "S1, dress rehearsal, stop";
- **whether per-case components are returned at all** — if validation reports only a
  scalar, the diagnostic table below is unavailable and each remaining slot degrades
  to a single-variable A/B yielding one bit each. Plan for both.
- **the per-component profile**, read against these local Task 1 values:

| component | local (Task 1) | what a gap localises |
|---|---|---|
| `decision_gate_pass_rate` | 0.7143 | the decision model does not generalise — dominates everything, since a gate failure zeroes the whole case |
| `mean_tool_score` | 0.9846 | reference reveal sets differ off-training; the reveal policy is mistuned |
| `mean_section_grounding_score` | 0.6747 | same cause, different symptom |
| `variable_weight_weighted_kappa` | 0.6045 | the weight vector is fitted to training annotators |
| `mean_rationale_score` | 0.8062 | text does not travel — but expect ≈ +0.118 from the judge offset, so only a gap *beyond* that counts |
| overall | 0.7207 | — |

A uniform sag across all six means the cohort is harder and nothing is wrong. A
single component collapsing is the actionable case, and it names its own fix.

### S2 — the response to S1's largest localised gap

The mapping is fixed **now**, so that S1's number cannot be rationalised into
whichever change we already felt like making:

| S1 shows | S2 submits |
|---|---|
| gate-pass rate down | Task 1 decision widened toward the majority class in the ambiguous high-PI-RADS/prior-positive leaf — that leaf holds 22 of 24 false-yes errors and splits 21/22, so off-training it is the only place the gate can be bleeding |
| tool or grounding down | a reveal-policy change. **This is the one place Route B comes back**: it was rejected at −0.0045 on *training* reference sets, and validation is the only instrument that can price it against real ones |
| weight-κ down | a flatter, less annotator-specific weight vector |
| rationale down beyond the offset | revert Items 8–9's gap clause, which is the newest and least-tested text |
| nothing gaps | **do not probe.** Go straight to the dress rehearsal and bank the remaining slots |

That last row is the important one. With an unknown cohort size and a one-shot test,
chasing a sub-point difference on unseen data is the same selecting-on-noise the Task 3
tie-break and the Item 8 stratum restriction were both refused for.

### S3 — second iteration on whatever S2 moved, or the second-largest gap

Only if S2's result is legible. If S2 came back inside noise, S3 is not spent.

### S4 — contingency, held

Reserved for the organizer's answer to Q2. If a deterministic MCP orchestrator is
ruled non-compliant, the LLM writer stops being optional and the resulting variant is
an architecture change that must be validated on a real cohort before Sep 10. If no
contingency materialises by **Sep 6**, S4 converts to a third probe.

### S5 — dress rehearsal, Sep 8. The exact frozen test artefact.

Submitted after the test configuration is frozen and changed in no way afterward.
Debug can prove the container runs, but only against cases from our own training data;
this is the only evidence we will ever have that the precise artefact going to the
one-shot test survives contact with an unseen cohort. Best-counts means it costs
nothing to spend a slot on a submission we already expect to score well.

If S5 fails, there are two days and the debug phase to recover in, which is exactly
why it is not scheduled for Sep 9.

## Stop rules

- **If S1 is within ~0.02 of (local − 0.118 on the rationale term) with no component
  gap**, stop probing. Freeze, dress-rehearse, write the paper.
- **If any slot's result is not legible** — cohort too small, components not returned,
  delta inside noise — do not spend the next slot trying to make it legible.
- **Turnaround is unmeasured.** If S1 takes more than ~24 h to return, drop S3 and
  keep the dress rehearsal; the slots are bounded by the calendar, not by the count.

## What is deliberately not being validated

- **Task 3.** It ranks on C-index alone and its reasoning trace is worth zero. The
  `cspca` tie-break is bounded below one CAPRA point by construction, so its
  worst case off-training is a return to plain CAPRA-S. Nothing to learn per slot.
- **Task 2.** The strongest task (CV 0.71 against 0.27 for a constant) and untouched
  since the refit. It carries the same weight as Task 1, so a Task 2 collapse in S1
  would be the biggest single finding available — but it is a *read* of S1, not a
  slot.
- **Confidence.** Proved constant-optimal under `1 − |Δ|/2` (0.643 always-`clear` vs
  0.633 `borderline` on the hard stratum). There is no variant to test.

  Aug 30 is the standing temptation to reopen this, and the note is here so it is
  reopened on evidence rather than on sting. Two of the three gate-passed Task 1 debug
  cases scored `confidence_score` **0.0** — `clear` against a reference `uncertain` is
  the *maximum* ordinal distance, so the component pays **nothing**, not half; at weight
  0.20 that is the single largest per-case loss visible anywhere in the run, and the
  judge docked the rationale for the same mismatch on top of it. It is still not a
  finding. Item 6's 0.736 is measured over all 91 labelled Task 1 cases and tied exactly
  by a per-stratum oracle; four cases drawn from that same cohort cannot overturn it, and
  the two that hurt are the tail the constant is already known to pay for.
