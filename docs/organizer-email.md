# Correspondence with the CHIMERA-agent organizers

**Letter 1 — sent 29 Aug.** Reproduced below unchanged, as the record of what was
asked. Answered since:

- **Q2 (does a deterministic MCP orchestrator qualify?)** — answered **no**, on
  8 Sep. Quoted in full in [`plan.md`](plan.md) and in the docstring of
  `chimera/predictors/acquire.py`; the operative sentence is that "a predefined
  deterministic retrieval strategy, such as an if/else policy that decides the
  sequence of sections in advance, would not meet the intended agentic setup".
  Implemented the same day — see Letter 2 below, which reports what complying
  cost and why the cost is a property of the metric rather than of our entry.
- **Q5 (is the judge live?)** — answered by the debug metrics dumps, not by the
  organizers; `mean_rationale_score` is non-null throughout. The question was
  left in the sent letter because the bug reports under it still stand.

**Letter 2 — drafted 8 Sep, unsent.** At the foot of this file.

---

## Letter 1 (sent 29 Aug)

To: nadieh.khalili@radboudumc.nl
Subject: CHIMERA-agent: questions on the MCP requirement and Task 1 grounding, plus evaluator and judge bug reports

---

Dear Dr Khalili and the CHIMERA-agent team,

Thank you for organising the challenge and for releasing the baseline and
evaluation code — having the scorer in the open has made it possible to build
against the real contract rather than guess at it.

We are preparing an entry targeting all three tasks and have a small number of
questions from working through `evaluate.py`, the released training data and two
debug-phase submissions. The first three concern scoring design and an
architectural decision we need to settle before validation opens on 1 September;
the next two are bug reports we think are worth acting on regardless of us. If
you have time for only one, question 2 is the one that gates our design.

**1. `cost_aware_tool_score` rewards declaring no tool use at all.**

The release notes explain that Task 2's `reveal_sequence` is "currently
unavailable and is represented as an empty list", which answers what we had
first meant to ask. But working out the consequence surfaced something we think
is a genuine scoring-design issue, and it is not specific to Task 2.

`cost_aware_tool_score` is precision — `|agent ∩ reference| / |agent|` — and
returns **1.0 when the agent declares no tools at all**, on the reasoning that no
unnecessary cost was incurred. That makes an empty `reveal_sequence` *weakly
dominant* on this component: it scores the maximum against any reference, and a
perfectly-chosen reveal set can only equal it, never beat it. An agent that
retrieves nothing is scored at least as well as one that retrieves exactly what
the urologist did.

`section_grounding_score` is the only counterweight, and it is bounded: variables
readable from the patient card (`psa`, `age`) and variables whose sections lie
outside the reveal vocabulary (`comorbidity`) are grounded or excluded for free.
An agent that weights *only* those is fully grounded having read nothing, and
collects both components at maximum.

We are flagging it rather than quietly exploiting it. If the intent is to reward
efficient-but-real retrieval, a recall term, or scoring an empty reveal set as
0.0 rather than 1.0 when the reference is non-empty, would close it. We are
happy to open an issue with a worked example.

**2. Does a deterministic MCP orchestrator satisfy the tool-use requirement?**

The rules state that the official MCP interface must be used for tool access. We
would like to confirm what that requires. Specifically: is it acceptable for the
*selection* of which sections to retrieve to be made by a deterministic policy,
provided every retrieval is then genuinely executed through the official MCP
server and the declared `reveal_sequence` reflects exactly those calls? Or is an
LLM-driven agent loop required, with the model itself choosing each tool call?

We are committed to the declared trace being truthful either way — our entry
already performs every retrieval over the official MCP interface, and a test
asserts that the emitted `reveal_sequence` is exactly the set of tools actually
invoked. The answer determines only whether the *selection* may be a learned
policy or must be a prompt, and since that is a substantial rebuild we would be
grateful for a steer before the validation window opens.

**3. Can `bx` ever be grounded on Task 1?**

`section_variable_mapping.json` maps `bx` to the pathology report, but Task 1
clinical-data payloads contain only `radiology_report`, `previous_notes`,
`psa_trend`, `laboratory_results` and `family_history` — there is no pathology
section to retrieve. A submission that weights `bx` as important or decisive on
Task 1 therefore appears to take an unavoidable `section_grounding_score`
penalty, unless it declares a reveal it could not have performed.

This is not hypothetical for us. The reference annotations weight `bx` on Task 1,
so matching them is worth more on `variable_weights` than the grounding penalty
costs, and our entry weights it `important` on every Task 1 case and simply
absorbs the loss. The alternative — declaring a `pathology_report` reveal to
recover the grounding — would mean reporting a retrieval we did not and could not
perform, which we are not willing to do.

Is the penalty intended, or should `bx` be treated as ungradable for grounding on
Task 1 in the way `comorbidity` already is? We note that release Version 3 also
removed `bx`, `bx_isup`, `bx_gl_prim` and `bx_gl_sec` from all 195 Task 1
structured prompts, so on the current release the variable is neither retrievable
nor present on the card, yet is still gradable.

**4. Two shapes make `evaluate.py` raise during aggregation, losing a whole task.**

`evaluate_case` keeps the *raw* prediction values when a case fails the schema
gate, but *normalised* values when it passes; the aggregator then does arithmetic
on whatever it finds. Two consequences, both of which take down the entire task
rather than the single bad case, because the exception escapes aggregation:

- **Task 3** — a case whose `months_to_recurrence` is a string *and* whose
  `event` is invalid: the unnormalised string survives into
  `aggregate_recurrence_metrics`, giving
  `TypeError: unsupported operand type(s) for -: 'str' and 'float'`.
- **Task 2** — a schema-failed case whose `treatment_recommendation.primary` is a
  truthy non-string: the raw value is copied into `pred_decision` and reaches
  scikit-learn, which refuses to mix string and numeric labels. (`""` and `None`
  are safe; they fall through to the missing-prediction sentinel.)

A participant tripping either loses a full task's score to an exception rather
than to a bad prediction. Normalising in the schema-failed branch of
`evaluate_case`, as the success branch already does, appears sufficient. We are
happy to open an issue or a pull request if that is useful.

**5. Three ways the rationale judge penalises rationales that are correct.**

The judge is clearly active on the leaderboard — `mean_rationale_score` is
non-null throughout the metrics dumps from both of our debug submissions — and at
a weight of 0.20 it is tied for the largest component of a Task 1/2 case score.
Reading its reason strings against our own outputs turned up three cases where it
penalises a rationale that is accurate. All three are in the evidence context it
is given rather than in the model backing it, so none looks like judge variance.

- **The judge cannot see the patient card.** `evaluate.py:1354` builds the
  evidence context from the clinical-data socket alone, so any value quoted from
  `structured-prompt.json` is unverifiable to it. On one Task 2 case we checked
  every value we cited against the inputs: those in the clinical data were
  accepted, and `age` and `bx_isup` — structured-prompt fields, quoted faithfully
  — were called unsupported. This penalises exactly the variables the task asks
  the model to reason about, and it is self-inflicted only in the sense that we
  could stop citing the card and write a vaguer rationale.

- **On Task 1 the judge attributes the reference's text to ours.** Two debug
  cases were docked for "hallucinating" an ISUP grade, e.g. *"while the Actual
  Output mentions 'ISUP grade group 1 (Gleason 3+3)', this specific grading is
  not present in the provided clinical data"*. We never wrote it — the string
  "ISUP" appears nowhere in our Task 1 output and cannot, since no Task 1 case
  carries a grade field. It appears in the **reference** rationale for both cases
  ("Again missing earlier ISUP grading…"). The judge appears to be reading the
  Expected Output and attributing it to the Actual Output.

- **On Task 3 the judge is given the answer and grades the prediction.**
  `evaluate.py:1315` puts `reference_months_to_recurrence` into the judge's
  input, and the reason strings then penalise our rationale for predicting a
  different figure. That is the task rather than a property of the rationale, and
  it means Task 3 rationale scores partly measure C-index error twice. (Task 3
  ranks on the C-index alone, so this costs us nothing — we raise it because it
  affects any entry whose reasoning trace is scored.)

We would be glad to supply the case IDs and full reason strings for any of these.

**6. Two quick confirmations.**

- The Task 1 output socket appears to have been renamed from
  `prostate-biospy-decision` to the corrected `prostate-biopsy-decision`. The
  debug phase now requires the corrected spelling, but the algorithm template
  Grand Challenge generates still writes the misspelling, as does the
  `predictions.json` fixture in the evaluation repo. Could you confirm the
  rename has been applied to the **validation and test phases** as well? We are
  currently emitting both filenames because we cannot inspect those phases, and
  the test phase allows only one submission.
- Neither `chimera-agent-baseline` nor `CHIMERA-agent` currently carries a
  licence file, which leaves them all-rights-reserved by default. Since the
  rules require submitted source to be "released under a permissive license",
  participants building on the baseline have no explicit permission to
  redistribute derived code, and prize eligibility additionally requires an
  open-source submission. Would you consider adding a permissive licence? We
  have written our own implementation against the published interface to avoid
  the question, but others may not.

**7. Manuscript logistics, which we could not find published.**

- The submission page requires a 6-page LNCS method description to qualify for
  the final ranking, but we can find **no manuscript deadline or submission
  channel** on the timeline, rules or submission pages — only the 10 September
  deadline for validation and test submissions. When is the paper due, and
  where should it be sent?
- The rules accept "a public or private GitHub URL", while the submission page
  offers prize incentives "exclusively for open-source submissions". Must the
  repository be **public at the time of submission**, or may a private repo be
  made public later without affecting prize eligibility?

Thank you for your time — we appreciate that this is a lot of detail, and are
glad to supply minimal reproductions for anything above.

With best regards,

Austin M Yunker
Loyola University Chicago

---

## Letter 2 (drafted 8 Sep — **unsent**)

To: nadieh.khalili@radboudumc.nl
Subject: CHIMERA-agent: we have implemented the MCP ruling — one measurement it turned up on Task 2

---

Dear Dr Khalili and the CHIMERA-agent team,

Thank you for the ruling on agentic retrieval. We have implemented it: as of
today our entry chooses each MCP call from the content the previous call
returned, so the sequence differs case to case and is decided during the case
rather than before it. Across the 163 labelled Task 1 and Task 2 training cases
it produces four distinct retrieval patterns on Task 1 and three on Task 2, and
no case reaches a decision without calling the server. Previously Task 2 made no
calls at all, which we agree was not in the spirit of the challenge.

We are writing because implementing it produced a measurement we think you will
want before the test phase closes. It concerns the released ground truth rather
than our entry, and we would raise it identically if it had gone the other way.

**On Task 2 the metric scores the required behaviour at exactly zero.**

Every released Task 2 reference rationale has an empty `reveal_sequence` — all
72 labelled cases, in both `train_release` and `train_release_v2` (144 files,
zero exceptions). Since `cost_aware_tool_score` is
`|agent ∩ reference| / |agent|`, any non-empty declaration on Task 2 has an
empty intersection and scores **0.0**, while declaring nothing scores **1.0**.

Running the released `evaluate.py` over our own before-and-after outputs, with
the decisions held byte-identical so the only variable is retrieval:

| Task 2, 72 cases        | before (no calls) | after (agentic) |
|-------------------------|-------------------|-----------------|
| `mean_tool_score`       | 1.0000            | **0.0000**      |
| `mean_section_grounding_score` | 0.2180     | 0.8910          |
| `mean_case_score`       | 0.6398            | 0.6121          |
| `ranking_score`         | 0.7444            | 0.7305          |

Grounding rises sharply, as it should — we now actually read the sections the
variables come from — but at the live component weights it recovers only about a
third of what the tool term takes away. Complying costs us $-0.0139$ of Task 2
ranking score and $-0.0216$ overall at the weights the leaderboard uses. We are
not asking for that back; we raise it because of what produces it.

**The consequence we think matters is the absent gradient, not the level.**

Because the numerator is zero for every non-empty declaration, *all* non-empty
declarations score identically on Task 2. A single, well-chosen call and an
indiscriminate sweep of all six sections receive the same 0.0 — and the sweep
then scores strictly higher on `section_grounding_score`. On Task 2 the metric
therefore has a mild preference for retrieving **everything** over retrieving
what the case needs, which we take to be the opposite of what a term named
`cost_aware` is for. An entry optimising the metric as released would either
declare nothing (now ruled out) or retrieve indiscriminately; targeted retrieval
is the one strategy that is dominated.

Task 1 shows a milder version of the same shape. There the references are
generous — 178 of 182 are non-empty and the modal set is four sections — so a
selective agent is penalised only for the calls it makes that the urologist did
not: our Task 1 `mean_tool_score` falls 0.9846 → 0.9128 purely from becoming
selective, with grounding unchanged at 0.6747. Since the component is precision
with no recall term, its maximum on Task 1 is still reached by retrieving as
little as possible.

**A fix with precedent inside your own scorer.** The release notes describe
Task 2's `reveal_sequence` as "currently unavailable", which we read as *not
recorded* rather than *recorded as empty*. If so, the natural treatment is the
one `section_grounding_score` already applies to `comorbidity`: drop the
component from the denominator when the reference cannot grade it, rather than
scoring it 0.0. That would leave Task 1 unchanged and make Task 2 neutral on
retrieval instead of adversarial to it. Adding a recall term, or scoring an
empty agent declaration as 0.0 against a non-empty reference, would address the
Task 1 half; we noted both in our August letter and they still apply.

We have kept our entry on the compliant behaviour regardless of how this is
resolved, and the empty declaration remains weakly dominant for anyone who has
not read the ruling. We would rather the metric did not reward that.

**One point of confirmation.** Our reading is that the `reveal_sequence` we
declare must be exactly the set of sections our MCP calls actually returned, and
that declaration *order* is not scored — `cost_aware_tool_score` compares sets.
Our declaration is in the order our agenda opened the questions, which is not
always the order the calls went out, because a section can be read twice and the
client memoises. If order is meant to carry information we will emit call order
instead; please say so and we will change it before the test submission.

Happy to supply case IDs, the two run directories, or a minimal reproduction of
the table above.

With best regards,

Austin M Yunker
Loyola University Chicago

---

### Notes for the sender, not part of the letter

- Numbers are the released `evaluate.py` with the judge off, so they are
  reproducible by the organizers without an API key. The $-0.0216$ overall is
  the live-weight (judge-on) repricing; the judge-off overall move is $-0.0071$,
  the difference being that grounding carries 0.175 with the judge off and 0.05
  with it on, so the grounding recovery is worth much less on the leaderboard
  than the table alone suggests. Say "at the weights the leaderboard uses" and
  do not quote $-0.0071$ unless asked, or the two figures will look inconsistent.
- The order question at the end is worth asking now rather than after the test
  phase: it is the one remaining way our declaration could be judged
  non-conforming, it costs nothing to change, and we cannot test it ourselves.
- Do **not** add the `reason`-field leak (Letter 1 §5 territory) to this letter.
  It is a separate report and pending its own reply; mixing a courtesy
  measurement with a security-shaped disclosure buries both.
