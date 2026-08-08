"""A fast, in-process replica of the official deterministic scorer.

Model selection needs to score a cohort thousands of times. The official
``evaluate.py`` needs a ``predictions.json``, a directory of per-case output
files, and -- with the judge on -- a running Ollama. This module computes the
same numbers from records already in memory, which is what makes nested CV and
the 64-subset reveal search in C3 tractable.

**The judge is out of scope here, permanently.** Everything below reproduces
the ``USE_RATIONALE_JUDGE=0`` path, where the evaluator drops the 0.10 rationale
weight and renormalises the remaining five components. An LLM judge is not
reproducible to 1e-9 and has no place in a selection loop.

Every function here mirrors one in ``evaluation/evaluate.py``, named the same
where possible, and ``tests/test_scorer_parity.py`` drives both over randomised
records to assert they agree. Read this file as a transcription, not as an
independent implementation -- when it disagrees with the official scorer, the
official scorer is right and this file is the bug.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Iterable, Sequence

from chimera.contract import spec
from chimera.scoring.records import TASK_KIND

# --------------------------------------------------------------------------- #
# Constants transcribed from evaluate.py
# --------------------------------------------------------------------------- #

#: Stand-in label for a case with no usable prediction. Cannot equal any real
#: label, so such a case costs the true class its recall instead of dropping
#: out of the F1 entirely.
MISSING_DECISION_LABEL = "__missing__"

#: The urologist form returns this when a row was never revealed.
_WEIGHT_ALIAS = {"not_revealed": "not_used"}

#: Horizons (months) for the reported cumulative/dynamic AUC.
TD_AUC_HORIZONS_MONTHS = (12.0, 24.0, 36.0, 60.0)

#: Task 1/2 case-score weights **with the rationale judge disabled**. With the
#: judge on these are 0.20 / 0.25 / 0.15 / 0.15 / 0.15 plus 0.10 for rationale.
CASE_COMPONENT_WEIGHTS: dict[str, float] = {
    "confidence": 0.225,
    "var_weight": 0.275,
    "factor_f1": 0.175,
    "tool": 0.150,
    "section_grounding": 0.175,
}

#: Task 3 case-score weights with the judge disabled (0.35/0.35/0.30 with it on).
#: Note the Task 3 case score does *not* feed the leaderboard -- C-index alone does.
RECURRENCE_COMPONENT_WEIGHTS: dict[str, float] = {"event": 0.50, "time": 0.50}

_MAX_CONF_DISTANCE = max(spec.CONFIDENCE_ORDINAL.values()) - min(spec.CONFIDENCE_ORDINAL.values())
_MAX_WEIGHT_DISTANCE = max(spec.WEIGHT_ORDINAL.values()) - min(spec.WEIGHT_ORDINAL.values())


# --------------------------------------------------------------------------- #
# Normalisers. The evaluator is lenient about casing and whitespace; we must be
# exactly as lenient, or CV will disagree with the leaderboard on sloppy input.
# --------------------------------------------------------------------------- #

def _norm_weight(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    v = _WEIGHT_ALIAS.get(v, v)
    return v if v in spec.WEIGHT_ORDINAL else None


def _norm_conf(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in spec.CONFIDENCE_ORDINAL else None


def _norm_decision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in spec.BIOPSY_DECISIONS else None


def _norm_treatment_decision(record: dict | None) -> str | None:
    if not isinstance(record, dict):
        return None
    rec = record.get("treatment_recommendation") or {}
    if not isinstance(rec, dict):
        return None
    value = rec.get("primary")
    if not isinstance(value, str):
        return None
    v = value.strip().lower().replace("-", "_")
    v = "_".join(v.split())
    return v if v in spec.TREATMENT_DECISIONS else None


def _norm_event(value: Any) -> int | None:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return None
    return iv if iv in (0, 1) else None


def _norm_months(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reveal_keys(record: dict) -> list[str]:
    """Revealed section names in order, deduplicated.

    Names outside the six-name vocabulary are kept *verbatim*, so inventing a
    section still counts as an extra reveal and costs tool precision.
    """
    seq = record.get("reveal_sequence") or []
    if not isinstance(seq, list):
        return []
    seen: set[str] = set()
    keys: list[str] = []
    for item in seq:
        if not isinstance(item, str) or not item or item in seen:
            continue
        keys.append(item)
        seen.add(item)
    return keys


def task_kind(record: dict) -> str:
    if "months_to_recurrence" in record:
        return "recurrence"
    if "treatment_recommendation" in record:
        return "treatment"
    return "biopsy"


def get_case_id(record: dict) -> str:
    cid = record.get("case_id")
    if cid:
        return str(cid)
    patient = record.get("patient") or {}
    if isinstance(patient, dict) and patient.get("id"):
        return str(patient["id"])
    return ""


def validate_record(record: Any, task: str) -> tuple[bool, str]:
    """The evaluator's schema gate. Failing it scores the case zero."""
    if not isinstance(record, dict):
        return False, "candidate is not an object"
    if task == "recurrence":
        if _norm_event(record.get("event")) is None:
            return False, f"invalid event={record.get('event')!r}"
        if _norm_months(record.get("months_to_recurrence")) is None:
            return False, f"invalid months_to_recurrence={record.get('months_to_recurrence')!r}"
        return True, "ok"
    if task == "treatment":
        decision = _norm_treatment_decision(record)
        if decision not in spec.TREATMENT_DECISIONS:
            raw = (record.get("treatment_recommendation") or {}).get("primary")
            return False, f"invalid treatment_recommendation.primary={raw!r}"
    else:
        decision = _norm_decision(record.get("biopsy_decision"))
        if decision not in spec.BIOPSY_DECISIONS:
            return False, f"invalid biopsy_decision={record.get('biopsy_decision')!r}"
    weights = record.get("variable_weights")
    if weights is not None and not isinstance(weights, dict):
        return False, "variable_weights must be an object"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Per-case components
# --------------------------------------------------------------------------- #

def decision_score(task: str, gt: dict, pred: dict) -> tuple[float, str | None, str | None]:
    """The hard gate: 1.0 on an exact match, 0.0 otherwise. No partial credit."""
    if task == "treatment":
        g, p = _norm_treatment_decision(gt), _norm_treatment_decision(pred)
    else:
        g, p = _norm_decision(gt.get("biopsy_decision")), _norm_decision(pred.get("biopsy_decision"))
    return (1.0 if (g == p and g is not None) else 0.0), g, p


def confidence_score(gt: dict, pred: dict) -> float | None:
    g, p = _norm_conf(gt.get("confidence")), _norm_conf(pred.get("confidence"))
    if g is None or p is None:
        return None
    distance = abs(spec.CONFIDENCE_ORDINAL[g] - spec.CONFIDENCE_ORDINAL[p])
    return 1.0 - (distance / _MAX_CONF_DISTANCE)


def variable_weight_score(gt: dict, pred: dict) -> float | None:
    """Mean ordinal error over the **ground-truth** variable keys.

    Iterating the ground truth rather than our own output has two consequences
    worth exploiting: a variable we omit is silently scored ``not_used``, and a
    variable we invent costs nothing *here* (though it still costs set-F1 and
    grounding).
    """
    gt_w = gt.get("variable_weights") or {}
    pr_w = pred.get("variable_weights") or {}
    if not isinstance(gt_w, dict) or not gt_w:
        return None
    if not isinstance(pr_w, dict):
        pr_w = {}
    errors: list[float] = []
    for var, gv in gt_w.items():
        g = _norm_weight(gv)
        if g is None:
            continue
        p = _norm_weight(pr_w.get(var, "not_used")) or "not_used"
        errors.append(
            abs(spec.WEIGHT_ORDINAL[g] - spec.WEIGHT_ORDINAL[p]) / _MAX_WEIGHT_DISTANCE
        )
    return (1.0 - mean(errors)) if errors else None


def _important_set(weights: Any) -> set[str]:
    if not isinstance(weights, dict):
        return set()
    return {var for var, val in weights.items() if _norm_weight(val) in spec.ACTIVE_WEIGHTS}


def _set_f1(gt_set: set[str], pred_set: set[str]) -> float:
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0
    tp = len(gt_set & pred_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    return 2 * precision * recall / (precision + recall)


def important_decisive_factor_score(gt: dict, pred: dict) -> float | None:
    gt_set = _important_set(gt.get("variable_weights") or {})
    pred_set = _important_set(pred.get("variable_weights") or {})
    if not gt_set and not pred_set:
        return 1.0
    return _set_f1(gt_set, pred_set)


def cost_aware_tool_score(gt: dict, pred: dict) -> float:
    """Precision of our reveals against the urologist's: ``|ours ∩ theirs| / |ours|``.

    Asymmetric on purpose -- extra reveals are charged, missing ones are free,
    and revealing nothing scores a perfect 1.0. Section grounding is the only
    thing pulling the other way.
    """
    expected = set(_reveal_keys(gt))
    actual = set(_reveal_keys(pred))
    if not expected and not actual:
        return 1.0
    if not actual:
        return 1.0
    return len(actual & expected) / len(actual)


def section_grounding_score(pred: dict) -> tuple[float | None, dict[str, list[str]]]:
    """Fraction of actively-weighted variables whose source section we revealed.

    A variable counts as grounded when it is always available (``psa``, ``age``),
    when it has no primary section at all, or when one of its primary sections
    is in our ``reveal_sequence``. A variable whose primary sections all fall
    outside the six-name reveal vocabulary -- ``comorbidity`` -- is *ungradable*
    and drops out of the denominator rather than being an unavoidable penalty.

    Returns ``(None, details)`` when nothing is weighted above ``not_used``,
    which the case composite then treats as a zero for this component.
    """
    revealed = set(_reveal_keys(pred))
    weights = pred.get("variable_weights") or {}
    if not isinstance(weights, dict):
        weights = {}

    grounded: list[str] = []
    ungrounded: list[str] = []
    ungradable: list[str] = []

    for var, weight_val in weights.items():
        w = _norm_weight(weight_val)
        if w is None or w == "not_used":
            continue
        if var in spec.ALWAYS_AVAILABLE_VARIABLES:
            grounded.append(var)
        elif var in spec.UNGRADABLE_FOR_GROUNDING:
            # Checked before the empty-tuple case below: the spec records an
            # empty tuple both for "no primary section" (grounded) and for
            # "primary section outside the vocabulary" (ungradable), and only
            # this set tells them apart.
            ungradable.append(var)
        else:
            primary = spec.PRIMARY_SECTIONS_BY_VARIABLE.get(var, ())
            if not primary or any(s in revealed for s in primary):
                grounded.append(var)
            else:
                ungrounded.append(var)

    details = {
        "grounded_variables": sorted(grounded),
        "ungrounded_variables": sorted(ungrounded),
        "ungradable_variables": sorted(ungradable),
    }
    total = len(grounded) + len(ungrounded)
    if total == 0:
        return None, details
    return len(grounded) / total, details


# --------------------------------------------------------------------------- #
# Task 3 components
# --------------------------------------------------------------------------- #

def recurrence_event_score(gt: dict, pred: dict) -> float | None:
    g, p = _norm_event(gt.get("event")), _norm_event(pred.get("event"))
    if g is None or p is None:
        return None
    return 1.0 if g == p else 0.0


def recurrence_time_score(gt: dict, pred: dict) -> float | None:
    """Censoring-aware closeness. Predicting *later* than a censoring time is free."""
    g_event = _norm_event(gt.get("event"))
    g_t = _norm_months(gt.get("months_to_recurrence"))
    p_t = _norm_months(pred.get("months_to_recurrence"))
    if g_t is None or p_t is None:
        return None
    scale = max(g_t, 1.0)
    if g_event == 1:
        return max(0.0, 1.0 - abs(p_t - g_t) / scale)
    if p_t >= g_t:
        return 1.0
    return max(0.0, 1.0 - (g_t - p_t) / scale)


def concordance_index(
    times: Sequence[float], preds: Sequence[float], events: Sequence[int]
) -> float | None:
    """Harrell's C-index, where a *shorter* predicted time means higher risk.

    This is the entire Task 3 leaderboard metric, so only the ordering of our
    predicted months matters -- their absolute scale is free.
    """
    num = 0.0
    den = 0.0
    n = len(times)
    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if i == j or not times[i] < times[j]:
                continue
            den += 1.0
            if preds[i] < preds[j]:
                num += 1.0
            elif preds[i] == preds[j]:
                num += 0.5
    return (num / den) if den > 0 else None


def _censoring_km(times: Sequence[float], events: Sequence[int]) -> list[tuple[float, float]]:
    """Kaplan-Meier estimate of G(t) = P(C > t), censoring treated as the event."""
    order = sorted(range(len(times)), key=lambda i: times[i])
    steps: list[tuple[float, float]] = [(float("-inf"), 1.0)]
    at_risk = len(order)
    g = 1.0
    i = 0
    while i < len(order):
        t = times[order[i]]
        tied = 0
        censored = 0
        while i + tied < len(order) and times[order[i + tied]] == t:
            if events[order[i + tied]] == 0:
                censored += 1
            tied += 1
        if at_risk > 0 and censored > 0:
            g *= 1.0 - censored / at_risk
        steps.append((t, g))
        at_risk -= tied
        i += tied
    return steps


def _km_at(steps: Sequence[tuple[float, float]], t: float) -> float:
    value = 1.0
    for step_t, step_g in steps:
        if step_t <= t:
            value = step_g
        else:
            break
    return value


def time_dependent_auc(
    times: Sequence[float], preds: Sequence[float], events: Sequence[int], horizon: float
) -> float | None:
    """Uno's IPCW cumulative/dynamic AUC at one horizon. Reported, not ranked."""
    steps = _censoring_km(times, events)
    cases: list[tuple[float, float]] = []
    controls: list[tuple[float, float]] = []
    for t, p, e in zip(times, preds, events):
        risk = -p
        if e == 1 and t <= horizon:
            g = _km_at(steps, t)
            if g > 0.0:
                cases.append((risk, 1.0 / g))
        elif t > horizon:
            g = _km_at(steps, horizon)
            if g > 0.0:
                controls.append((risk, 1.0 / g))
    if not cases or not controls:
        return None

    num = 0.0
    for case_risk, case_w in cases:
        for ctrl_risk, ctrl_w in controls:
            if case_risk > ctrl_risk:
                num += case_w * ctrl_w
            elif case_risk == ctrl_risk:
                num += 0.5 * case_w * ctrl_w
    den = sum(w for _, w in cases) * sum(w for _, w in controls)
    return (num / den) if den > 0 else None


# --------------------------------------------------------------------------- #
# Per-case scoring
# --------------------------------------------------------------------------- #

def _recurrence_row(gt: dict, pred: dict | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": get_case_id(gt),
        "task": "recurrence",
        "gate": "passed",
        "case_score": 0.0,
        "gt_event": _norm_event(gt.get("event")),
        "pred_event": None,
        "gt_months": _norm_months(gt.get("months_to_recurrence")),
        "pred_months": None,
        "event_score": None,
        "time_score": None,
        "rationale_score": None,
    }
    if pred is None:
        row["gate"] = "missing_candidate"
        return row

    ok, _ = validate_record(pred, "recurrence")
    if not ok:
        row["gate"] = "schema_failed"
        row["pred_event"] = pred.get("event")
        row["pred_months"] = pred.get("months_to_recurrence")
        return row

    es = recurrence_event_score(gt, pred)
    ts = recurrence_time_score(gt, pred)
    row["pred_event"] = _norm_event(pred.get("event"))
    row["pred_months"] = _norm_months(pred.get("months_to_recurrence"))
    row["event_score"] = es
    row["time_score"] = ts

    w = RECURRENCE_COMPONENT_WEIGHTS
    score = (es or 0.0) * w["event"] + (ts or 0.0) * w["time"]
    row["case_score"] = max(0.0, min(1.0, score))
    return row


def score_case(gt: dict, pred: dict | None) -> dict[str, Any]:
    """Score one case. Mirrors ``evaluate_case`` with the judge disabled.

    The returned row carries every component separately, which is what the C3
    selector optimises against.
    """
    task = task_kind(gt)
    if task == "recurrence":
        row = _recurrence_row(gt, pred)
        _attach_kappa_fields(row, gt, pred)
        return row

    gt_decision = (
        _norm_treatment_decision(gt) if task == "treatment" else _norm_decision(gt.get("biopsy_decision"))
    )
    row: dict[str, Any] = {
        "case_id": get_case_id(gt),
        "task": task,
        "gate": "passed",
        "case_score": 0.0,
        "decision_score": 0.0,
        "decision_correct": False,
        "gt_decision": gt_decision,
        "pred_decision": None,
        "confidence_score": None,
        "variable_weight_score": None,
        "important_decisive_factor_score": None,
        "tool_score": None,
        "section_grounding_score": None,
        "rationale_score": None,
    }

    if pred is None:
        row["gate"] = "missing_candidate"
        _attach_kappa_fields(row, gt, None)
        return row

    ok, _ = validate_record(pred, task)
    if not ok:
        row["gate"] = "schema_failed"
        # Asymmetry copied from the evaluator: a schema-failed *treatment* case
        # carries its raw (invalid) decision into the dataset F1, while a
        # schema-failed *biopsy* case falls through to the missing sentinel.
        if task == "treatment":
            rec = pred.get("treatment_recommendation") or {}
            row["pred_decision"] = rec.get("primary") if isinstance(rec, dict) else None
        _attach_kappa_fields(row, gt, pred)
        return row

    ds, gt_decision, pred_decision = decision_score(task, gt, pred)
    row["decision_score"] = ds
    row["gt_decision"] = gt_decision
    row["pred_decision"] = pred_decision
    row["decision_correct"] = ds == 1.0

    if ds == 0.0:
        # The hard gate. Every component below is skipped and the case is worth
        # exactly zero, however good the reasoning was.
        row["gate"] = f"{task}_decision_failed"
        _attach_kappa_fields(row, gt, pred)
        return row

    cs = confidence_score(gt, pred)
    vws = variable_weight_score(gt, pred)
    fs = important_decisive_factor_score(gt, pred)
    ts = cost_aware_tool_score(gt, pred)
    sgs, _details = section_grounding_score(pred)

    row["confidence_score"] = cs
    row["variable_weight_score"] = vws
    row["important_decisive_factor_score"] = fs
    row["tool_score"] = ts
    row["section_grounding_score"] = sgs

    w = CASE_COMPONENT_WEIGHTS
    score = (
        (cs or 0.0) * w["confidence"]
        + (vws or 0.0) * w["var_weight"]
        + (fs or 0.0) * w["factor_f1"]
        + (ts or 0.0) * w["tool"]
        + (sgs or 0.0) * w["section_grounding"]
    )
    row["case_score"] = max(0.0, min(1.0, score))
    _attach_kappa_fields(row, gt, pred)
    return row


def _attach_kappa_fields(row: dict, gt: dict, pred: dict | None) -> None:
    """Raw label pairs the dataset-level kappas need."""
    row["gt_biopsy_decision_conf"] = _norm_conf(gt.get("confidence"))
    row["pred_biopsy_decision_conf"] = _norm_conf(pred.get("confidence")) if pred else None
    pairs: list[tuple[int, int]] = []
    if pred and (row.get("decision_score") or 0.0) > 0.0:
        gt_w = gt.get("variable_weights") or {}
        pr_w = pred.get("variable_weights") or {}
        if not isinstance(pr_w, dict):
            pr_w = {}
        for var, gv in gt_w.items():
            g = _norm_weight(gv)
            if g is None:
                continue
            p = _norm_weight(pr_w.get(var, "not_used")) or "not_used"
            pairs.append((spec.WEIGHT_ORDINAL[g], spec.WEIGHT_ORDINAL[p]))
    row["_weight_pairs"] = pairs


# --------------------------------------------------------------------------- #
# Cohort aggregation
# --------------------------------------------------------------------------- #

def _aggregate_recurrence(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    evaluated = [r for r in rows if r.get("pred_event") is not None and r.get("pred_months") is not None]
    event_pairs = [
        (r["gt_event"], r["pred_event"])
        for r in rows
        if r.get("gt_event") is not None and r.get("pred_event") is not None
    ]
    event_scores = [r["event_score"] for r in rows if r.get("event_score") is not None]
    time_scores = [r["time_score"] for r in rows if r.get("time_score") is not None]
    maes = [
        abs(r["pred_months"] - r["gt_months"])
        for r in rows
        if r.get("gt_event") == 1 and r.get("pred_months") is not None and r.get("gt_months") is not None
    ]

    times: list[float] = []
    preds: list[float] = []
    events: list[int] = []
    for r in rows:
        if r.get("gt_months") is not None and r.get("pred_months") is not None and r.get("gt_event") is not None:
            times.append(r["gt_months"])
            preds.append(r["pred_months"])
            events.append(r["gt_event"])

    td_auc = {f"{int(h)}m": time_dependent_auc(times, preds, events, h) for h in TD_AUC_HORIZONS_MONTHS}
    td_auc_values = [v for v in td_auc.values() if v is not None]
    c_index = concordance_index(times, preds, events)

    return {
        "n_cases": n,
        "n_evaluated": len(evaluated),
        "mean_case_score": mean(r["case_score"] for r in rows),
        # Task 3 ranks on the C-index alone; the case score is analysis only.
        "ranking_score": c_index,
        "recurrence_event_accuracy": (mean(int(a == b) for a, b in event_pairs) if event_pairs else None),
        "mean_event_score": (mean(event_scores) if event_scores else None),
        "mean_time_score": (mean(time_scores) if time_scores else None),
        "event1_time_mae_months": (mean(maes) if maes else None),
        "concordance_index": c_index,
        "time_dependent_auc": td_auc,
        "mean_time_dependent_auc": (mean(td_auc_values) if td_auc_values else None),
        "mean_rationale_score": None,
    }


def _aggregate_decision(rows: list[dict]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    n = len(rows)
    is_treatment = all(r.get("task") == "treatment" for r in rows)

    graded = [r for r in rows if r.get("gt_decision") is not None]
    y_true = [r["gt_decision"] for r in graded]
    y_pred = [r.get("pred_decision") or MISSING_DECISION_LABEL for r in graded]
    n_correct = sum(int(r.get("decision_score") == 1.0) for r in graded)

    conf_pairs = [
        (spec.CONFIDENCE_ORDINAL[r["gt_biopsy_decision_conf"]],
         spec.CONFIDENCE_ORDINAL[r["pred_biopsy_decision_conf"]])
        for r in rows
        if r.get("gt_biopsy_decision_conf") in spec.CONFIDENCE_ORDINAL
        and r.get("pred_biopsy_decision_conf") in spec.CONFIDENCE_ORDINAL
    ]

    flat_gt_w: list[int] = []
    flat_pred_w: list[int] = []
    for r in rows:
        for gw, pw in r.get("_weight_pairs", []):
            flat_gt_w.append(gw)
            flat_pred_w.append(pw)

    gate_pass = [r for r in rows if r.get("decision_score", 0.0) > 0.0]
    tool_scores = [r["tool_score"] for r in rows if r["tool_score"] is not None]
    grounding = [r["section_grounding_score"] for r in rows if r["section_grounding_score"] is not None]

    out: dict[str, Any] = {
        "n_cases": n,
        "n_evaluated": sum(1 for r in graded if r.get("pred_decision") is not None),
        "n_decision_correct": n_correct,
        "n_decision_incorrect": len(graded) - n_correct,
        "mean_case_score": mean(r["case_score"] for r in rows),
        "ranking_score": None,
        "decision_accuracy": None,
        "decision_f1_yes": None,
        "decision_weighted_f1": None,
        "confidence_weighted_kappa": None,
        "variable_weight_weighted_kappa": None,
        "mean_tool_score": mean(tool_scores) if tool_scores else None,
        "mean_section_grounding_score": mean(grounding) if grounding else None,
        "mean_rationale_score": None,
        "decision_gate_pass_rate": len(gate_pass) / n,
        "mean_case_score_among_gate_passed": mean(r["case_score"] for r in gate_pass) if gate_pass else 0.0,
    }

    if y_true:
        out["decision_accuracy"] = float(accuracy_score(y_true, y_pred))
        if is_treatment:
            try:
                out["decision_weighted_f1"] = float(f1_score(
                    y_true, y_pred,
                    labels=sorted(spec.TREATMENT_DECISIONS),
                    average="weighted", zero_division=0,
                ))
            except Exception:
                out["decision_weighted_f1"] = None
        else:
            try:
                out["decision_f1_yes"] = float(f1_score(
                    y_true, y_pred, labels=["yes"], average=None, zero_division=0,
                )[0])
            except Exception:
                out["decision_f1_yes"] = None

    if conf_pairs and len({*(c for c, _ in conf_pairs), *(p for _, p in conf_pairs)}) >= 2:
        try:
            ct, cp = zip(*conf_pairs)
            k = float(cohen_kappa_score(list(ct), list(cp), weights="quadratic"))
            out["confidence_weighted_kappa"] = k if k == k else None
        except Exception:
            out["confidence_weighted_kappa"] = None

    if flat_gt_w and len(set(flat_gt_w) | set(flat_pred_w)) >= 2:
        try:
            k = float(cohen_kappa_score(flat_gt_w, flat_pred_w, weights="quadratic"))
            out["variable_weight_weighted_kappa"] = k if k == k else None
        except Exception:
            out["variable_weight_weighted_kappa"] = None

    # The leaderboard number: mean case score and the task's decision F1,
    # weighted equally.
    task_f1 = out["decision_weighted_f1"] if is_treatment else out["decision_f1_yes"]
    if task_f1 is not None:
        out["ranking_score"] = (out["mean_case_score"] + task_f1) / 2.0
    return out


def score_cohort(cases: Iterable[tuple[dict, dict | None]]) -> dict[str, Any]:
    """Score a whole cohort and return the aggregate metrics.

    ``cases`` pairs each ground-truth record with its prediction record, or
    ``None`` where no prediction exists. Pass the pairs in the same order the
    official run would produce them (scored jobs first, then unmatched ground
    truth) if you want byte-identical floats; every statistic here is in fact
    order-independent, but matching the order keeps diffs readable.

    The returned dict has the same keys as the official ``aggregate_metrics``,
    with ``mean_rationale_score`` pinned to ``None``.
    """
    rows = [score_case(gt, pred) for gt, pred in cases]
    if not rows:
        return {"n_cases": 0}
    if all(r.get("task") == "recurrence" for r in rows):
        return _aggregate_recurrence(rows)
    return _aggregate_decision(rows)


def score_cohort_rows(cases: Iterable[tuple[dict, dict | None]]) -> tuple[list[dict], dict[str, Any]]:
    """``score_cohort`` but keeping the per-case rows, for error analysis."""
    rows = [score_case(gt, pred) for gt, pred in cases]
    if not rows:
        return rows, {"n_cases": 0}
    if all(r.get("task") == "recurrence" for r in rows):
        return rows, _aggregate_recurrence(rows)
    return rows, _aggregate_decision(rows)
