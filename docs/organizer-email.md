# Draft: email to the CHIMERA-agent organizers

To: nadieh.khalili@radboudumc.nl
Subject: CHIMERA-agent: questions on the MCP requirement, Task 2 reveal annotations, and two evaluator crashes

---

Dear Dr Khalili and the CHIMERA-agent team,

Thank you for organising the challenge and for releasing the baseline and
evaluation code — having the scorer in the open has made it possible to build
against the real contract rather than guess at it.

We are preparing an entry targeting all three tasks and have a small number of
questions from working through `evaluate.py` and the released training data. The
first three affect architectural decisions we need to make now; the fourth is a
bug report we think is worth acting on regardless of us.

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

We are committed to the declared trace being truthful either way — we do not
intend to report reveals we did not perform — but the answer determines whether
our evidence-selection component can be a learned policy or must be a prompt.

**3. Can `bx` ever be grounded on Task 1?**

`section_variable_mapping.json` maps `bx` to the pathology report, but Task 1
clinical-data payloads contain only `radiology_report`, `previous_notes`,
`psa_trend`, `laboratory_results` and `family_history` — there is no pathology
section to retrieve. A submission that weights `bx` as important or decisive on
Task 1 therefore appears to take an unavoidable `section_grounding_score`
penalty, unless it declares a reveal it could not have performed. Is that
intended, or should `bx` be treated as ungradable for grounding on Task 1 in the
way `comorbidity` already is?

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

**5. Is the LLM rationale judge active on the leaderboard?**

`evaluate.py` gates `mean_rationale_score` behind `USE_RATIONALE_JUDGE`, which
changes the case-score weights materially — 0.10 to the judge, with the five
deterministic components renormalised down. Our debug-phase score came back
noticeably above what we compute offline with the judge disabled, which we cannot
fully account for from case mix alone.

Could you confirm whether the judge runs in the validation and test phases, and
if so which model backs it? We would like our offline scoring to match the
leaderboard's, and the answer changes how much of our effort should go to the
free-text rationale versus the structured fields.

**6. Three quick confirmations.**

- The Task 1 output socket appears to have been renamed from
  `prostate-biospy-decision` to the corrected `prostate-biopsy-decision`. The
  debug phase now requires the corrected spelling, but the algorithm template
  Grand Challenge generates still writes the misspelling, as does the
  `predictions.json` fixture in the evaluation repo. Could you confirm the
  rename has been applied to the **validation and test phases** as well? We are
  currently emitting both filenames because we cannot inspect those phases, and
  the test phase allows only one submission.
- Neither `chimera-agent-baseline` nor `CHIMERA-agent` currently carries a
  licence file, which leaves them all-rights-reserved by default. Since ranked
  entries must publish a public repository, participants building on the
  baseline have no explicit permission to redistribute derived code. Would you
  consider adding a permissive licence? We have written our own implementation
  against the published interface to avoid the question, but others may not.

Thank you for your time — we appreciate that this is a lot of detail, and are
glad to supply minimal reproductions for anything above.

With best regards,

Austin M Yunker
Loyola University Chicago
