# CHIMERA-agent (MICCAI 2026)

Entry for the [CHIMERA-agent challenge](https://chimera-agent.grand-challenge.org/):
prostate-cancer decision support across three tasks, where every prediction must
be accompanied by a structured reasoning trace that is itself scored.

Full strategy, checkpoints and timeline: [`docs/plan.md`](docs/plan.md).

## Status

**C0 — contract conformance: complete.** The official evaluator scores our
output end to end, on all three tasks, offline.

| Task | Ranking metric | Constant-predictor floor |
|---|---|---|
| 1 | `(mean_case_score + F1_yes) / 2` | 0.569 on the 3 reference cases |
| 2 | `(mean_case_score + weighted F1) / 2` | 0.444 on the 3 reference cases |
| 3 | Harrell's C-index | 0.500 on a 60-case synthetic cohort |

Those are floors from a fixed-output predictor, not results. Task 3's 0.500 is
the correct value for constant predictions (every pair ties).

**C1 — scorer parity: complete.** `chimera.scoring.fast` reproduces the
judge-disabled evaluator exactly, so cross-validation no longer pays for a
subprocess per fold. Agreement is asserted to 1e-9 at two levels:

- **The maths** — both scorers driven over the same in-memory records, on
  randomised hostile cohorts (invalid tokens, wrong types, absent predictions,
  every-case-fails-the-gate, all-censored) and on the reference cases.
- **The whole path** — a run directory built by `run_local`, scored by the real
  `evaluate.py` as a subprocess, and diffed against the fast scorer reading the
  *same directory* back off disk. Nothing is shared between the two sides but
  the files.

```bash
./scripts/score.sh work/run/constant                              # official
python -m chimera.cli.score_fast --run work/run/constant --compare  # + diff
```

**C1b — submission container: built and smoke-tested.** The image builds in CI,
runs offline under Grand Challenge's mounts, and every output passes the
evaluator's schema gate. Payload was a *fitted* constant prior, superseded by C2
below.

**C2 — decision models: complete.** Guideline strata with **metric-fitted leaf
labels**: the partition comes from clinical knowledge, and each leaf's decision is
chosen by maximising the official ranking score — not accuracy, which is a
different objective. Task 3 uses **CAPRA-S**, a published post-prostatectomy
nomogram, with nothing fitted at all.

Scored by the **official evaluator** on the real labels, with repeated pooled
out-of-fold cross-validation alongside:

| Task | C1b prior | C2 | CV (5×5) | constant baseline (CV) |
|---|---|---|---|---|
| 1 | 0.635 | 0.698 | 0.690 ± 0.000 | 0.635 ± 0.000 |
| 2 | 0.292 | **0.713** | **0.708 ± 0.002** | 0.273 ± 0.038 |
| 3 | 0.500 | **0.737** | 0.737 (nothing fitted) | 0.500 |

Fitted and scored on **release Version 3**. Overall in-sample 0.716.

**Task 1 got there by dropping a leaf, not adding one.** The first version gave
prior-positive-biopsy cases their own stratum — the clinically obvious move,
since imaging genuinely does not discriminate in men with known cancer (21 yes
against 22 no at high PI-RADS). Out of fold it cost 0.034 and made the fitted
labels unstable across splits (0.648 ± 0.022 against 0.690 ± 0.000 for PI-RADS
alone). Six patients change and PI-RADS alone is right on four of them.

Three findings drove the wins:

- **Task 2 is a two-stage decision, not a flat four-way choice.** Prior biopsy
  result separates `continued_surveillance` almost perfectly (13 of 14 negative
  biopsies), and EAU risk then splits the rest — `positive_high` is 12/12
  `active_treatment`. Gate pass went 0.431 → 0.806.
- **Task 3's reports are templated**, so every surgical-pathology field CAPRA-S
  needs parses at 100%: grade, stage, margins, EPE, SVI, and nodal status with
  pNx correctly distinguished from pN0.
- **The tool score is precision, not recall** — `|declared ∩ pathologist| /
  |declared|`, so a section read and declared for a feature no leaf consults is a
  straight subtraction, and a section missed costs nothing. Once Task 1 stopped
  consulting prior-biopsy status it stopped reading the notes for it, and its
  mean tool score went 0.770 → 0.946.

Reveal honesty runs the other way too: whatever the feature extractor *does*
read has to appear in `reveal_sequence`. `extract_structured` reports the
sections it consumed, the fit forces them into the reveal policy, and the
predictor unions them per case in case the parameter file is older than the code.

The image cannot be built or run here — no Docker, and rootless podman is
unavailable — so it is built **and smoke-tested** in GitHub Actions:

```bash
# What CI does, and the only place the container ever executes:
docker build --platform=linux/amd64 --tag chimera-agent .
scripts/smoke_test_image.sh chimera-agent work/fixtures work/smoke

# The entrypoint itself runs natively, no container required:
CHIMERA_INPUT=work/fixtures/task1/BX_01 CHIMERA_OUTPUT=/tmp/out python inference.py
```

Pushing a `v*` tag builds the image, smoke-tests it, asserts the fitted
parameters are reachable from inside it, and uploads the `docker save` tarball as
a workflow artifact. `v0.2.1` is the current C2 image: ~47.6 MB gzipped, all steps green.

Two crashes in the official evaluator were found along the way; both are pinned
by tests so we never emit the shapes that trigger them — see
[Evaluator traps](#evaluator-traps-we-must-not-step-in).

## Setup

```bash
conda create -y -n chimera python=3.11
conda activate chimera
pip install -e '.[dev]'

# Reference repos. Cloned, never vendored — re-clone to pick up upstream changes.
git clone --depth 1 https://github.com/DIAGNijmegen/chimera-agent-baseline refs/baseline
git clone --depth 1 https://github.com/DIAGNijmegen/CHIMERA-agent          refs/challenge
```

## Workflow

```bash
# 1. Cohorts. Fixtures mirror the 8 reference ground-truth cases; the synthetic
#    cohort is larger and carries its own labels for exercising the metrics.
python -m chimera.cli.make_fixtures
python -m chimera.cli.make_synthetic_cohort

# 2. Run a predictor over a cohort.
python -m chimera.cli.run_local --cases work/fixtures --out work/run/constant

# 3. Score with the OFFICIAL evaluator. This is the only source of reported numbers.
./scripts/score.sh work/run/constant

# Against the synthetic cohort (needs its ground truth):
CHIMERA_GT_ROOT="$PWD/work/synth/ground_truth" \
  ./scripts/score.sh work/run/synth-constant

# 4. Score in-process, and check it against the official run above.
python -m chimera.cli.score_fast --run work/run/constant --compare

# 5. Tests. CHIMERA_REQUIRE_REFS=1 makes a skip a failure, as CI does.
pytest
```

### The real cohort

The release ships each case's inputs and labels in one directory with no
`inputs.json`, so it needs splitting before `read_case` can consume it.

```bash
# cases/ (inputs only) + ground_truth/ (labels + section mapping)
python -m chimera.cli.make_release_cases --data data/train_release --out work/train

# Refit the prior; re-run whenever labels grow.
python -m chimera.cli.fit_prior --data data/train_release

python -m chimera.cli.run_local --cases work/train/cases --predictor prior \
                                --out work/run/prior
CHIMERA_GT_ROOT="$PWD/work/train/ground_truth" ./scripts/score.sh work/run/prior
```

### Two cohorts, two purposes

`work/fixtures/` matches the eight ground-truth cases in the challenge repo. It
is the only cohort with *real* labels, so it is the one that tells the truth —
but it is far too small to measure anything, and both its Task 3 cases are
censored, which leaves the C-index undefined.

`work/synth/` is 180 generated cases with generated labels. It exists to drive
every metric code path (all four treatment classes, comparable survival pairs,
IPCW horizons) at a realistic cohort size. **Never report a number measured on
it and never tune against it.**

Both are regenerated deterministically and are gitignored. Delete
`work/fixtures/` and point at the real cohort once the training data lands.

## Layout

```
inference.py   Grand Challenge entrypoint (paths env-overridable for local runs)
Dockerfile     python:3.11-slim — the entrypoint's imports are stdlib-only
src/chimera/
  contract/    spec.py — every slug, filename, enum and vocabulary, in one place
               types.py — typed predictions + pre-write validation
               io.py — GC socket reader / writer
               aggregate.py — rebuilds predictions.json for the evaluator
  evidence/    structured.py — the patient card, typed; reports.py — templated
               report parsing; notes.py — prior-biopsy status read back out of
               the referral prose, for when the card stops carrying it. All take
               a CaseInputs, so training and serving share the feature code.
  models/      guidelines.py — CAPRA-S, EAU risk, the Task 1 partition
               stratified.py — leaf-label and reasoning fitting
  eval/        cv.py — repeated pooled out-of-fold CV (dev only, not shipped)
  predictors/  base.py — the one-method Predictor protocol
               constant.py — the C0 floor
               prior.py + prior_params.json — the fitted C1b payload
               guideline.py + guideline_params.json — the C2 payload
  scoring/     fast.py — judge-free transcription of the official scorer
               records.py — the flat record shape both scorers compare on,
                            plus the inverse of the predictions.json writer
  cli/         make_fixtures · make_synthetic_cohort · make_release_cases ·
               run_local · score_fast · fit_prior · fit_models ·
               cross_validate · check_outputs
tests/         contract conformance · spec-vs-evaluator agreement ·
               scorer parity (in-memory) · run-directory parity (through files) ·
               entrypoint + prior + stdlib-only import closure
scripts/       score.sh — the official evaluator, run natively
               smoke_test_image.sh — run the image as GC does, then validate
.github/       ci.yml (tests, skips are failures) · build-image.yml (build,
               smoke-test, docker save artifact)
refs/          reference repos (gitignored)
data/          released challenge data (gitignored; CC BY-NC-SA 4.0)
work/          cohorts and run artefacts (gitignored)
```

### Licensing

Apache-2.0 (`LICENSE`, `NOTICE`). Note that **both DIAGNijmegen reference repos
are unlicensed** — no `LICENSE` file and no GitHub license metadata — so they are
all-rights-reserved and nothing may be copied from them. `refs/` is gitignored
and everything in `src/` is written against the published interface rather than
derived from their code; socket slugs, filenames and enum vocabularies are
factual interface details, not expression.

The **released data is CC BY-NC-SA 4.0** — non-commercial, share-alike,
attribution required. It stays in gitignored `data/` and is never redistributed
here. Nothing in `src/` embeds it, so the share-alike term does not reach this
repository's Apache-2.0 grant; a published adaptation of the data itself would
be a different question.

## Contract details that cost silent zeros

Each of these is enforced by a test in `tests/`.

- **The Task 1 socket was misspelled `biospy` and has been corrected.** The
  2026-08-24 debug submission rejected the misspelling — *"Output file
  'prostate-biopsy-decision.json' was not produced"* — so the corrected name is
  canonical. We still write the old one alongside it: sockets are configured per
  phase, only the debug phase has been observed, and the test phase is a single
  submission with no retry. See `spec.LEGACY_OUTPUT_FILENAMES`.
- **Grand Challenge truncates socket slugs to 50 characters** but not
  `relative_path`. Identify interfaces by slug, resolve files by path.
- **Decision files are bare JSON values** — `"yes"`, not `{"decision": "yes"}`.
- **Task 1/2 reasoning is a flat object with exactly four keys**; Task 3's
  reasoning socket is a **bare string**. The baseline's `Task1Output` Pydantic
  models are a richer *internal* shape and will not validate if submitted.
- **`case_id` must be inlined** in the `structured-prompt` input value of
  `predictions.json`. Without it the evaluator skips the job and scores the
  ground-truth case as a missing candidate worth zero.
- **The rationale judge reads clinical context from the inline clinical-data
  input**, because ground truth no longer carries it.
- **A missing or schema-invalid prediction is scored against a sentinel label**,
  so skipping a hard case costs the true class its recall. Never drop a case;
  degrade to a conservative prediction instead.

## Evaluator traps we must not step in

Two shapes make `evaluate.py` raise rather than score, and an exception in
aggregation loses **the entire task**, not the one bad case. Both come from the
same root cause: `evaluate_case` stores *raw* prediction values when a case
fails the schema gate, but *normalised* ones when it passes, and the aggregator
then does arithmetic on whatever it finds.

| Task | Shape | What happens |
|---|---|---|
| 3 | `months_to_recurrence` is a string **and** `event` is invalid | `TypeError: unsupported operand type(s) for -: 'str' and 'float'` |
| 2 | `treatment_recommendation.primary` is a truthy non-string | sklearn refuses to mix string and numeric labels |

Our writers coerce and validate before anything reaches disk, so we cannot emit
either. `tests/test_contract.py` pins both — those tests fail loudly if a future
evaluator fixes the bug, which is the signal to delete them. Worth reporting
upstream.

## Environment notes

- **No Docker on this host** (only `singularity-ce`; `/etc/subuid` is empty, so
  rootless podman/buildah are unavailable). Submission images are built in
  GitHub Actions and downloaded as a `docker save` artefact.
- **The V100s are compute capability 7.0**, and vLLM 0.25 — the baseline's pin —
  dropped Volta. Local LLM work goes through an OpenAI-compatible endpoint
  (llama.cpp / Ollama); the container targets **A10G (sm_86)**.
- The official evaluator normally ships as a Docker image with an embedded
  Ollama judge. `scripts/score.sh` runs the identical `evaluate.py` natively
  instead. Only the judge transport differs.
- Keep model weights and caches on local scratch — the NFS home is 98% full.

> Per the challenge rules, only the official evaluation pipeline may be used to
> report performance. The fast in-process scorer (C1) is for cross-validation
> only and is verified against `evaluate.py`, never substituted for it.
