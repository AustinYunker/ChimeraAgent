"""The C2 decision model: guideline strata with metric-fitted labels.

Fitting has two stages, and both are cheap enough to be exact:

1. **Leaf labels.** Each guideline stratum from :mod:`chimera.models.guidelines` is
   assigned a decision. The candidate space is tiny -- 2^4 for Task 1, 4^5 for Task 2
   -- so it is enumerated exhaustively against the *official ranking metric* rather
   than against accuracy. That distinction is not pedantic: Task 1 scores
   ``(mean_case_score + F1_yes) / 2``, so a leaf that splits 26/23 against "yes" can
   still be worth labelling "yes" because of what the F1 term does.
2. **Reasoning constants, conditioned on the decision.** The C1b prior emitted one
   confidence and one weight vector for every case. Here they are fitted per
   decision, which costs nothing at inference and captures real structure.

   The conditioning set matters. A case only earns reasoning points when its decision
   is *correct*, so the constants attached to decision ``d`` are fitted on the
   training cases whose **true** label is ``d`` -- those are exactly the cases that
   will pass the gate when we predict ``d``. Fitting on cases we merely predict ``d``
   for would optimise against cases scoring zero regardless.

The reasoning fit reuses :func:`chimera.cli.fit_prior.fit` unchanged. Given a subset
whose labels are all ``d``, that optimiser selects ``d`` on its own -- predicting
anything else fails the gate on every row -- so no special-casing is needed.

Task 3 has no leaves and nothing fitted: CAPRA-S is published, and only the ordering
of the predicted months affects the C-index.
"""

from __future__ import annotations

from typing import Any, Sequence

from chimera.contract import spec
from chimera.contract.io import CaseInputs
from chimera.evidence import extract_reports, extract_structured
from chimera.mcp.client import ClinicalStore
from chimera.models.guidelines import LEAVES_BY_TASK, capra_s, stratum

#: Predicted months for a case whose risk score is unreadable. Mid-range, so an
#: unscoreable case is ranked neither best nor worst.
FALLBACK_MONTHS = 60.0

#: Below this, a decision keeps the pooled reasoning constants. Fitting a confidence
#: level and an 11-way weight vector on a handful of rows is noise, not structure.
#: `watchful_waiting` has 2 examples in 72 and never clears it.
MIN_ROWS_FOR_CONDITIONAL_FIT = 8

#: Maps a CAPRA-S score onto predicted months. Only the ordering matters for the
#: C-index, so this is an arbitrary decreasing map chosen to stay in a plausible
#: clinical range and to keep `months_to_recurrence` positive and finite.
MONTHS_AT_ZERO_RISK = 120.0
MONTHS_PER_CAPRA_POINT = 8.0


# --------------------------------------------------------------------------- #
# Records the scorer compares on
# --------------------------------------------------------------------------- #

def decision_record(task: int, case_id: str, decision: str, reasoning: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": case_id,
        "confidence": reasoning.get("confidence"),
        "variable_weights": reasoning.get("variable_weights") or {},
        "reveal_sequence": list(reasoning.get("reveal_sequence") or []),
    }
    if task == 1:
        record["biopsy_decision"] = decision
    else:
        record["treatment_recommendation"] = {"primary": decision}
    return record


def true_decision(task: int, gt: dict[str, Any]) -> str | None:
    if task == 1:
        value = gt.get("biopsy_decision")
        return value if isinstance(value, str) else None
    rec = gt.get("treatment_recommendation") or {}
    value = rec.get("primary") if isinstance(rec, dict) else None
    return value if isinstance(value, str) else None


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

def _score_leaf_map(
    task: int,
    rows: Sequence[Any],
    mapping: dict[str, str],
    reasoning: dict[str, Any],
    row_leaves: Sequence[str],
) -> float:
    from chimera.scoring.fast import score_cohort

    paired = []
    for row, leaf in zip(rows, row_leaves):
        decision = mapping[leaf]
        paired.append((row.gt, decision_record(task, row.case_id, decision, reasoning)))
    metrics = score_cohort(paired)
    return metrics.get("ranking_score") or 0.0


#: Above this many candidate label assignments, hill-climb instead of enumerating.
#: Task 1 has 2^4 = 16 and stays exact; Task 2 has 4^5 = 1024, and at ~11 ms per
#: cohort scoring that is 12 s per fold, or five minutes across a repeated CV run.
MAX_EXHAUSTIVE_LEAF_MAPS = 64


def fit_leaf_labels(
    task: int, rows: Sequence[Any], reasoning: dict[str, Any]
) -> dict[str, str]:
    """Argmax over label assignments, against the ranking metric rather than accuracy.

    The distinction is the point of this function. Task 1's metric is
    ``(mean_case_score + F1_yes) / 2``, so a leaf splitting 26/23 in favour of "no"
    can still be worth labelling "yes" once the F1 term is accounted for -- and only
    a search against the real metric will find that.

    Unlike the reasoning fit this genuinely needs the cohort-level scorer, because
    F1 couples all the leaves together and cannot be decomposed per case.
    """
    import itertools

    leaves = LEAVES_BY_TASK[task]
    decisions = spec.BIOPSY_DECISIONS if task == 1 else spec.TREATMENT_DECISIONS

    # Precompute each row's leaf once; stratum() is called O(candidates x rows) below.
    row_leaves = [stratum(task, extract_structured(r.case, r.store)) for r in rows]
    present = set(row_leaves)
    seen = [leaf for leaf in leaves if leaf in present]

    counts: dict[str, int] = {}
    for row in rows:
        label = true_decision(task, row.gt)
        if label:
            counts[label] = counts.get(label, 0) + 1
    default = max(counts, key=lambda k: counts[k]) if counts else decisions[0]

    # Seed: the majority label within each leaf. Sensible, and the starting point
    # for hill-climbing when the space is too large to enumerate.
    per_leaf: dict[str, dict[str, int]] = {leaf: {} for leaf in seen}
    for leaf, row in zip(row_leaves, rows):
        label = true_decision(task, row.gt)
        if label:
            per_leaf[leaf][label] = per_leaf[leaf].get(label, 0) + 1
    seed = {
        leaf: (max(c, key=lambda k: c[k]) if c else default) for leaf, c in per_leaf.items()
    }
    for leaf in leaves:
        seed.setdefault(leaf, default)

    def score(mapping: dict[str, str]) -> float:
        return _score_leaf_map(task, rows, mapping, reasoning, row_leaves)

    best_map, best_score = dict(seed), score(seed)

    if len(decisions) ** len(seen) <= MAX_EXHAUSTIVE_LEAF_MAPS:
        for combo in itertools.product(decisions, repeat=len(seen)):
            mapping = {**seed, **dict(zip(seen, combo))}
            value = score(mapping)
            if value > best_score:
                best_map, best_score = mapping, value
        return best_map

    for _ in range(6):
        improved = False
        for leaf in seen:
            for decision in decisions:
                if best_map[leaf] == decision:
                    continue
                mapping = {**best_map, leaf: decision}
                value = score(mapping)
                if value > best_score + 1e-12:
                    best_map, best_score, improved = mapping, value, True
        if not improved:
            break
    return best_map


def _mean_case_score(task: int, gts: Sequence[dict], reasoning: dict[str, Any]) -> float:
    """Mean case score with the gate assumed passed.

    Each case is scored as if we predicted its *true* decision, because reasoning
    only earns points on cases that clear the gate. Deliberately calls
    :func:`~chimera.scoring.fast.score_case` rather than ``score_cohort``: the
    cohort-level path runs scikit-learn for F1s and kappas that are irrelevant to a
    fixed decision, and it is ~9x slower per call. At roughly 10^5 calls per fit
    that is the difference between minutes and an hour.
    """
    from chimera.scoring.fast import score_case

    total = 0.0
    for gt in gts:
        decision = true_decision(task, gt)
        if decision is None:
            continue
        record = decision_record(task, str(gt.get("case_id", "")), decision, reasoning)
        total += score_case(gt, record)["case_score"]
    return total / len(gts) if gts else 0.0


def required_reveals(rows: Sequence[Any]) -> list[str]:
    """Sections the feature extractor reads, so every candidate reveal must name them.

    Since release Version 3 the Task 1 patient card no longer carries ``bx`` and
    :func:`chimera.evidence.extract_structured` recovers it from the radiology
    report and the referral notes instead. Reading a section obliges declaring it,
    so those sections are not optional at serve time -- and if the fit were free to
    leave them out, it would be optimising a reveal set the container cannot emit.

    Taken as a union over the cohort. On Task 1 that is exact: all 195 cases lack
    ``bx`` and all 195 carry both sections, so every case reads the same two. Where
    a cohort is mixed the union slightly overstates the reveal for the minority,
    which costs a little fidelity in the fit and nothing in honesty -- the serve
    path declares only what the individual case actually yielded.
    """
    from chimera.evidence import extract_structured

    seen: set[str] = set()
    for row in rows:
        seen.update(extract_structured(row.case, row.store).evidence_sections)
    return [s for s in spec.REVEAL_SECTIONS if s in seen]


def fit_reasoning_group(
    task: int, gts: Sequence[dict], sections: Sequence[str],
    required: Sequence[str] = (),
) -> dict[str, Any]:
    """Reasoning constants maximising the mean case score over ``gts``.

    Confidence is optimised separately from the rest: the case score is a weighted
    sum whose confidence term depends on nothing else, so searching it jointly with
    the weight vector would multiply the work by three for no gain.

    The weight vector and reveal set *are* coupled -- section grounding asks which
    actively-weighted variables were revealed -- so those are searched together and
    then hill-climbed one variable at a time.

    ``required`` sections appear in every candidate reveal set; see
    :func:`required_reveals`.
    """
    import itertools

    from chimera.cli.fit_prior import modal_weights

    variables = spec.VARIABLES_BY_TASK[task]
    # Canonical order, and drawn from `required` rather than from `sections`: a
    # section we read is declared whether or not the cohort scan happened to list it.
    forced = [s for s in spec.REVEAL_SECTIONS if s in set(required)]
    optional = [s for s in sections if s not in set(required)]
    reveal_sets = [
        forced + list(combo)
        for size in range(len(optional) + 1)
        for combo in itertools.combinations(optional, size)
    ]
    strategies = [{v: level for v in variables} for level in spec.WEIGHT_LEVELS]
    strategies.append(modal_weights(list(gts), variables))

    best = {
        "confidence": spec.CONFIDENCE_LEVELS[0],
        "variable_weights": strategies[0],
        "reveal_sequence": [],
    }
    best_score = float("-inf")
    for weights, reveals in itertools.product(strategies, reveal_sets):
        candidate = {
            "confidence": best["confidence"],
            "variable_weights": weights,
            "reveal_sequence": reveals,
        }
        score = _mean_case_score(task, gts, candidate)
        if score > best_score:
            best, best_score = candidate, score

    # Hill-climb the weight vector, re-checking reveals after each pass since
    # grounding depends on which variables are active.
    for _ in range(4):
        improved = False
        for var in variables:
            for level in spec.WEIGHT_LEVELS:
                weights = dict(best["variable_weights"])
                if weights.get(var) == level:
                    continue
                weights[var] = level
                candidate = {**best, "variable_weights": weights}
                score = _mean_case_score(task, gts, candidate)
                if score > best_score + 1e-12:
                    best, best_score, improved = candidate, score, True
        for reveals in reveal_sets:
            candidate = {**best, "reveal_sequence": list(reveals)}
            score = _mean_case_score(task, gts, candidate)
            if score > best_score + 1e-12:
                best, best_score, improved = candidate, score, True
        if not improved:
            break

    # Confidence last and separately -- its term is additive and independent.
    for level in spec.CONFIDENCE_LEVELS:
        candidate = {**best, "confidence": level}
        score = _mean_case_score(task, gts, candidate)
        if score > best_score + 1e-12:
            best, best_score = candidate, score

    return best


def fit_reasoning_by_decision(
    task: int, rows: Sequence[Any], sections: Sequence[str],
    required: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Reasoning constants per decision, fitted on the cases that will be gated in.

    Falls back to the pooled fit for a decision with too few examples --
    ``watchful_waiting`` has 2 cases in 72 and cannot support its own constants.

    ``required`` defaults to :func:`required_reveals` over ``rows``, so callers get
    the honest constraint without having to know it exists.
    """
    if required is None:
        required = required_reveals(rows)
    gts = [r.gt for r in rows]
    by_decision: dict[str, dict[str, Any]] = {
        "__default__": fit_reasoning_group(task, gts, sections, required)
    }

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = true_decision(task, row.gt)
        if label:
            groups.setdefault(label, []).append(row.gt)

    for decision, subset in groups.items():
        if len(subset) < MIN_ROWS_FOR_CONDITIONAL_FIT:
            continue
        by_decision[decision] = fit_reasoning_group(task, subset, sections, required)
    return by_decision


def fit(task: int, rows: Sequence[Any], sections: list[str]) -> dict[str, Any]:
    """Fit the whole Task 1/2 pipeline: reasoning constants, then leaf labels.

    Order matters. Leaf labels are chosen against the ranking metric, and that metric
    depends on the reasoning attached to each case, so the reasoning has to exist
    first. The pooled constants are used while searching labels; the conditional ones
    are applied at prediction time.
    """
    required = required_reveals(rows)
    reasoning = fit_reasoning_by_decision(task, rows, sections, required)
    leaf_labels = fit_leaf_labels(task, rows, reasoning["__default__"])
    return {
        "task": task,
        "leaf_labels": leaf_labels,
        "reasoning": reasoning,
        "required_reveals": required,
    }


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def reasoning_for(params: dict[str, Any], decision: str) -> dict[str, Any]:
    reasoning = params.get("reasoning") or {}
    return reasoning.get(decision) or reasoning.get("__default__") or {}


def predict_decision(
    case: CaseInputs, store: ClinicalStore, params: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """The decision for ``case`` and the reasoning constants that go with it."""
    task = params["task"]
    features = extract_structured(case, store)
    leaf = stratum(task, features)
    labels = params.get("leaf_labels") or {}
    decisions = spec.BIOPSY_DECISIONS if task == 1 else spec.TREATMENT_DECISIONS
    decision = labels.get(leaf)
    if decision not in decisions:
        decision = next(iter(labels.values()), decisions[0])
    return decision, reasoning_for(params, decision)


def predict_record(
    case: CaseInputs, store: ClinicalStore, params: dict[str, Any]
) -> dict[str, Any]:
    """Flat scorer-shaped record, for the CV harness."""
    task = params["task"]
    decision, reasoning = predict_decision(case, store, params)
    return decision_record(task, case.case_id, decision, reasoning)


# --------------------------------------------------------------------------- #
# Task 3
# --------------------------------------------------------------------------- #

def predict_months(case: CaseInputs, store: ClinicalStore) -> float:
    """Predicted months to recurrence, ordered by CAPRA-S. Nothing is fitted."""
    score = capra_s(extract_reports(store), extract_structured(case, store).psa)
    if score is None:
        return FALLBACK_MONTHS
    months = MONTHS_AT_ZERO_RISK - MONTHS_PER_CAPRA_POINT * score
    return max(1.0, months)


def predict_recurrence_record(
    case: CaseInputs, store: ClinicalStore, params: Any = None
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "months_to_recurrence": predict_months(case, store),
        # The event flag does not affect the C-index; the cohort is 56/75 censored.
        "event": 0,
    }
