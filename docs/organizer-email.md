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

**1. Task 2 `reveal_sequence` is empty in all 72 labeled training cases.**

Every labeled Task 2 case in `train_release` has `"reveal_sequence": []`, whereas
Task 1 averages three to four sections per case. Because `cost_aware_tool_score`
is precision against the reference reveal set, this makes *any* declared reveal
on Task 2 score zero on that component, and revealing nothing score 1.0.

Is that a genuine finding — Task 2 decisions were made without section
retrieval — or was that part of the annotation form not administered for Task 2?
We would rather not build a policy around an artifact. If the reference
annotations are likely to change before the test set, we would very much
appreciate knowing, since the two cases imply opposite optimal behaviour for
0.15 of the Task 2 case score.

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

**5. Three quick confirmations.**

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
