# Running the rationale judge locally

`scripts/score.sh` scores with `USE_RATIONALE_JUDGE=0`. That is deliberate — the
deterministic components are the reproducible ones, and the judge needs a 9.6 GB
model, a server, and a dependency tree that has no business near the predictors.

But `free_text` is **0.20 of the case score** on Tasks 1 and 2 since upstream
`192c39c`, so leaving it unmeasured means guessing about the largest single
reasoning component. `scripts/score-judged.sh` runs the same official evaluator
with the judge switched on. This file records how the pieces it expects were
installed.

Everything lives **outside the repo**, under `/home/beams0/AYUNKER/opt`: the model
store alone is ~9.0 GB, and none of it should ever be a `git clean` away from
deletion. Override with `OLLAMA_ROOT` if you put it elsewhere.

```
/home/beams0/AYUNKER/opt/
  ollama/          2.2G   the ollama 0.32.15 tarball, unpacked
  ollama-models/   9.0G   OLLAMA_MODELS — where `gemma4:e4b` lands
  ollama-logs/            serve.log, so a failed start is diagnosable
  judgeenv/        448M   deepeval + sklearn, for evaluate.py only
```

## What the judge actually is

`evaluate.py` builds a DeepEval `GEval` metric over a local Ollama model. Three
environment variables drive it, and only the third has a usable default here:

| variable | evaluator default | why it needs overriding |
|---|---|---|
| `USE_RATIONALE_JUDGE` | `1` | `scripts/score.sh` forces `0` |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | the compose service name; does not resolve outside the container |
| `JUDGE_MODEL` | `gemma4:e4b` | correct as-is — pull exactly this |

## Install

**Ollama.** A user-local unpack, not a system install — there is no root here and
no container runtime on this host.

```bash
mkdir -p ~/opt/ollama ~/opt/ollama-models ~/opt/ollama-logs
curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst -o /tmp/ollama.tar.zst
# This host's tar has no --zstd support (`tar --help | grep -c zstd` → 0), so pipe it.
zstd -dc /tmp/ollama.tar.zst | tar -x -C ~/opt/ollama
```

**The judge venv.** Separate from the `chimera` conda env on purpose: `deepeval`
pulls a large tree, and `tests/test_entrypoint.py` guards the predictors against
exactly that kind of dependency creep. It holds the evaluator's own
`docker/requirements.txt` and nothing else.

```bash
/home/beams/AYUNKER/miniconda3/envs/chimera/bin/python -m venv ~/opt/judgeenv
~/opt/judgeenv/bin/pip install -r refs/challenge/evaluation/docker/requirements.txt
```

Installed at the time of writing: deepeval 4.2.0, scikit-learn 1.9.0, numpy 2.4.6.

**The model.** `scripts/score-judged.sh` pulls it on first use, because
`evaluate.py` otherwise does so silently and a 9.6 GB wait looks like a hang.

## Use

```bash
CHIMERA_GT_ROOT="$PWD/work/train/ground_truth" \
  ./scripts/score-judged.sh work/run/<some-run>
```

The wrapper starts the server if it is not already up (`setsid`, so it outlives
the script) and sets `OLLAMA_KEEP_ALIVE=2h` so the next run reuses the resident
weights instead of paying the ~40 s load again.

All 43 layers offload to CUDA0 on the A6000s; context is 131072. Throughput is
roughly **15 s per case**, so a full 423-case run is around 1¾ hours. Run one
evaluation at a time — two concurrent runs contend for the single resident copy
of the model and both get slower.

Score a run **into its own directory**. `scripts/score.sh` writes `_scores/metrics.json`,
and `chimera.cli.score_fast --compare` checks the fast scorer against that file;
a judged `metrics.json` written over a judge-free one breaks the parity check,
which is why `work/run/judged-baseline` exists as a copy of `work/run/guideline-v3`
rather than as a re-score of it.

## Reproducibility

It is more reproducible than expected. `work/run/judged-rationale` and
`work/run/judged-nodre` differ only in the Task 1 rationales; their Task 2 and
Task 3 numbers came back **identical to four decimal places** — rationale 0.8362
and 0.2053 in both — over 72 and 75 cases. So on this host, same model, same
text, same score.

That is a property of this setup, not a guarantee. Do not lean on it across an
Ollama or model upgrade, and do not use judged numbers as a **parity** signal:
`tests/test_scorer_parity.py` and `chimera.cli.score_fast` stay judge-free by
design, and the deterministic components are unaffected by `free_text` anyway.
What it does buy is that an A/B of two rationale variants is a clean read — a
difference in the judged score is a difference in the text, not sampling noise.

## Calibration against the platform's judge

Reproducible is not the same as *correct*. The 12 debug cases give a free paired
check, because the platform scored them under `v0.2.1` and only `spec.py` and docs
have changed since — so `work/run/judged-baseline` carries byte-identical
`free_text`. Confirmed directly: the platform's reason string for
`PT-pseudo_0020cfca66c8` quotes `"Weighted most heavily: pirads, psa, age, bx"`,
which is what our baseline emits for that case.

Per-case `rationale_score`, platform against local, 11 cases (one gate failure):

| | value |
|---|---|
| platform mean | 0.500 |
| local mean | **0.382** |
| mean difference | **−0.118** (95% CI −0.223 … −0.013) |
| mean absolute difference | 0.155 |
| max absolute difference | 0.500 (T2-001: platform 0.9, local 0.4) |
| Pearson *r* | **+0.818** (p = 0.002) |
| Spearman *ρ* | **+0.831** (p = 0.002) |

**Our judge is a conservative proxy: it orders cases the way the platform's does,
and scores them about 0.12 lower.** Both halves matter.

The correlation is what licenses the instrument. We use it to *rank variants*, and
on ordering the two agree strongly — so the direction of a measured A/B is
trustworthy, which is the property the rewrite measurement actually leaned on.

The offset means absolute local scores must not be quoted as platform predictions,
and there is a sharper consequence: scores are quantised to 0.1 and capped at 1.0,
so a systematic **shift** cannot continue to hold near the top of the range. Task 2
post-rewrite scores 0.8621 locally; adding 0.118 puts it at 0.98, which is not
credible. Either the offset compresses as scores rise, or it is not a pure shift.
Both readings imply the **+0.0114 overall is an upper bound on what the platform
will show**, with the Task 2 component the most likely to shrink. That is a
hypothesis, not a measurement — n = 11, quantised to 0.1, and the CI on the bias
only barely excludes zero.

Do not chase the offset. Tuning local prose against a 0.12 gap on eleven cases is
fitting the proxy, not the target.

## Stopping the server

Not with `pkill -f "ollama serve"` — the pattern matches the shell running it and
kills the caller too (learned the hard way; exit code 144).

```bash
pkill -x ollama
```
