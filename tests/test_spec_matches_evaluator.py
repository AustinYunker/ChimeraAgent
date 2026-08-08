"""Assert our contract constants against the official evaluator's own constants.

``chimera.contract.spec`` is a hand-transcription of things buried in
``evaluate.py``. Transcriptions rot. These tests make the transcription
self-verifying: re-clone ``refs/`` and any upstream change to a slug, an ordinal
scale, a component weight, or the reveal vocabulary fails here.
"""

from __future__ import annotations

import pytest

from chimera.contract import spec


def test_interface_keys_match(official_evaluator):
    ev = official_evaluator
    assert spec.INTERFACE_KEY_BY_TASK[1] == ev.INTERF0_KEY
    assert spec.INTERFACE_KEY_BY_TASK[2] == ev.INTERF1_KEY
    assert spec.INTERFACE_KEY_BY_TASK[3] == ev.INTERF2_KEY
    assert set(spec.TASK_BY_INTERFACE_KEY) == set(ev.INTERFACE_TASK_ID)


def test_task_ids_agree(official_evaluator):
    ev = official_evaluator
    for key, task_id in ev.INTERFACE_TASK_ID.items():
        assert spec.TASK_BY_INTERFACE_KEY[key] == int(task_id.removeprefix("task"))


def test_output_slugs_are_accepted_by_the_evaluator(official_evaluator):
    """Whatever we write must be a slug the evaluator will look for."""
    ev = official_evaluator
    assert spec.OUTPUT_SOCKETS[1]["decision"][0] in ev.TASK1_DECISION_SLUGS
    assert spec.OUTPUT_SOCKETS[1]["reasoning"][0] in ev.TASK1_REASONING_SLUGS
    assert spec.OUTPUT_SOCKETS[2]["decision"][0] in ev.TASK2_DECISION_SLUGS
    assert spec.OUTPUT_SOCKETS[2]["reasoning"][0] in ev.TASK2_REASONING_SLUGS
    assert spec.OUTPUT_SOCKETS[3]["decision"][0] in ev.TASK3_OUTCOME_SLUGS
    assert spec.OUTPUT_SOCKETS[3]["reasoning"][0] in ev.TASK3_REASONING_SLUGS


def test_clinical_slugs_match(official_evaluator):
    ev = official_evaluator
    assert spec.CLINICAL_SLUG_BY_TASK[1] in ev.TASK1_CLIN_SLUGS
    assert spec.CLINICAL_SLUG_BY_TASK[2] in ev.TASK2_CLIN_SLUGS
    assert spec.CLINICAL_SLUG_BY_TASK[3] in ev.TASK3_CLIN_SLUGS


def test_decision_vocabularies_match(official_evaluator):
    ev = official_evaluator
    assert set(spec.BIOPSY_DECISIONS) == set(ev.VALID_BIOPSY_DECISIONS)
    assert set(spec.TREATMENT_DECISIONS) == set(ev.VALID_TREATMENT_DECISIONS)


def test_ordinal_scales_match(official_evaluator):
    """Our ordinals must agree in *order*, since scores are ordinal distances."""
    ev = official_evaluator
    assert spec.CONFIDENCE_ORDINAL == ev.CONF_MAP
    assert spec.WEIGHT_ORDINAL == ev.WEIGHT_MAP
    assert set(spec.ACTIVE_WEIGHTS) == set(ev.IMPORTANT_OR_DECISIVE)


def test_reveal_vocabulary_matches(official_evaluator):
    ev = official_evaluator
    assert set(spec.REVEAL_SECTIONS) == set(ev.SECTION_KEY_TO_REVEAL_NAME.values())


def test_primary_sections_match_the_section_mapping(official_evaluator):
    """Our variable -> reveal-section map must match the shipped mapping file."""
    ev = official_evaluator
    mapping = ev._get_section_var_mapping()
    if not mapping:
        pytest.skip("section_variable_mapping.json not resolvable from the evaluator")

    var_to_sections = mapping["variable_to_sections"]
    always = set(mapping["always_available_variables"]["variables"])
    assert always == set(spec.ALWAYS_AVAILABLE_VARIABLES)

    for var, info in var_to_sections.items():
        expected = tuple(
            ev.SECTION_KEY_TO_REVEAL_NAME[s]
            for s in info.get("primary_sections", [])
            if s in ev.SECTION_KEY_TO_REVEAL_NAME
        )
        assert spec.PRIMARY_SECTIONS_BY_VARIABLE[var] == expected, (
            f"{var}: ours={spec.PRIMARY_SECTIONS_BY_VARIABLE[var]} theirs={expected}"
        )


def test_ungradable_variables_are_identified(official_evaluator):
    """A variable with no reachable primary section can never be grounded.

    Weighting one above 'not_used' is free -- it is excluded from the grounding
    score rather than counted against us -- so the selector must know which
    these are.
    """
    ev = official_evaluator
    mapping = ev._get_section_var_mapping()
    if not mapping:
        pytest.skip("section_variable_mapping.json not resolvable")

    ungradable = set()
    for var, info in mapping["variable_to_sections"].items():
        primary = info.get("primary_sections", [])
        if not primary or info.get("always_available_baseline"):
            continue
        if var in mapping["always_available_variables"]["variables"]:
            continue
        if not any(s in ev.SECTION_KEY_TO_REVEAL_NAME for s in primary):
            ungradable.add(var)

    assert ungradable == set(spec.UNGRADABLE_FOR_GROUNDING)


def test_variable_sets_cover_the_ground_truth(official_evaluator):
    """Our per-task variable list must cover every key the ground truth weights.

    ``variable_weight_score`` iterates over ground-truth keys and scores a
    missing prediction as ``not_used``, so an omission is a silent penalty.
    """
    import json
    from pathlib import Path

    gt_root = Path(official_evaluator.__file__).parent / "ground_truth"
    for task, variables in spec.VARIABLES_BY_TASK.items():
        stem = "prostate-biopsy-decision" if task == 1 else "prostate-treatment-decision"
        files = sorted((gt_root / f"task{task}").glob(f"*/{stem}-reasoning.json"))
        if not files:
            pytest.skip(f"no ground-truth reasoning for task{task}")
        for path in files:
            gt_keys = set(json.loads(path.read_text())["variable_weights"])
            assert gt_keys <= set(variables), (
                f"{path.parent.name}: ground truth weights {sorted(gt_keys - set(variables))} "
                f"which task{task} does not emit"
            )
