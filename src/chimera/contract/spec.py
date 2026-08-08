"""The challenge I/O contract, in one place.

Every constant here was read off the official repos rather than the challenge
web pages, which do not document the submission shapes:

* socket slugs / relative paths -- ``evaluation/test/input/predictions.json``
  and ``inference.py`` in ``DIAGNijmegen/chimera-agent-baseline``
* scored field vocabularies     -- ``evaluation/evaluate.py`` and
  ``evaluation/ground_truth/section_variable_mapping.json``

Two spellings matter and are easy to get wrong:

* The Task 1 output socket is misspelled **biospy**. The evaluator accepts
  either spelling, but ``predictions.json`` declares ``biospy``, so that is
  what we write.
* Grand Challenge truncates socket *slugs* to 50 characters, which clips the
  Task 3 slugs. The ``relative_path`` is *not* truncated. Resolve files by
  ``relative_path``, identify interfaces by slug.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- #
# Interfaces. A job's task is the sorted tuple of its input socket slugs.
# --------------------------------------------------------------------------- #

STRUCTURED_PROMPT_SLUG: Final = "structured-prompt"
NEURAL_REP_SLUG: Final = "prostate-modality-level-neural-representations"

# The clinical-data slug is what distinguishes the three interfaces. The Task 3
# entry is the 50-char truncation of
# "prostate-time-to-recurrence-or-last-follow-up-clinical-data".
CLINICAL_SLUG_BY_TASK: Final[dict[int, str]] = {
    1: "prostate-biopsy-decision-clinical-data",
    2: "prostate-treatment-decision-clinical-data",
    3: "prostate-time-to-recurrence-or-last-follow-up-clin",
}
TASK_BY_CLINICAL_SLUG: Final[dict[str, int]] = {v: k for k, v in CLINICAL_SLUG_BY_TASK.items()}

INTERFACE_KEY_BY_TASK: Final[dict[int, tuple[str, ...]]] = {
    task: tuple(sorted((STRUCTURED_PROMPT_SLUG, NEURAL_REP_SLUG, clinical)))
    for task, clinical in CLINICAL_SLUG_BY_TASK.items()
}
TASK_BY_INTERFACE_KEY: Final[dict[tuple[str, ...], int]] = {
    v: k for k, v in INTERFACE_KEY_BY_TASK.items()
}

# --------------------------------------------------------------------------- #
# Output sockets. (slug, relative_path) per task -- slug truncated, path not.
# --------------------------------------------------------------------------- #

OUTPUT_SOCKETS: Final[dict[int, dict[str, tuple[str, str]]]] = {
    1: {
        "decision": ("prostate-biospy-decision", "prostate-biospy-decision.json"),
        "reasoning": (
            "prostate-biospy-decision-reasoning",
            "prostate-biospy-decision-reasoning.json",
        ),
    },
    2: {
        "decision": ("prostate-treatment-decision", "prostate-treatment-decision.json"),
        "reasoning": (
            "prostate-treatment-decision-reasoning",
            "prostate-treatment-decision-reasoning.json",
        ),
    },
    3: {
        "decision": (
            "prostate-time-to-recurrence-or-last-follow-up",
            "prostate-time-to-recurrence-or-last-follow-up.json",
        ),
        "reasoning": (
            # 50-char truncation of "...-follow-up-reasoning".
            "prostate-time-to-recurrence-or-last-follow-up-reas",
            "prostate-time-to-recurrence-or-last-follow-up-reasoning.json",
        ),
    },
}

# --------------------------------------------------------------------------- #
# Scored vocabularies.
# --------------------------------------------------------------------------- #

BIOPSY_DECISIONS: Final[tuple[str, ...]] = ("yes", "no")

TREATMENT_DECISIONS: Final[tuple[str, ...]] = (
    "active_surveillance",
    "continued_surveillance",
    "watchful_waiting",
    "active_treatment",
)

# Ordinal, worst -> best. The evaluator scores 1 - |distance| / 2.
CONFIDENCE_LEVELS: Final[tuple[str, ...]] = ("uncertain", "borderline", "clear")
CONFIDENCE_ORDINAL: Final[dict[str, int]] = {c: i for i, c in enumerate(CONFIDENCE_LEVELS)}

# Ordinal, least -> most. The evaluator scores 1 - mean(|distance|) / 3.
WEIGHT_LEVELS: Final[tuple[str, ...]] = ("not_used", "noted", "important", "decisive")
WEIGHT_ORDINAL: Final[dict[str, int]] = {w: i for i, w in enumerate(WEIGHT_LEVELS)}

# Counted by the important/decisive set-F1 component.
ACTIVE_WEIGHTS: Final[frozenset[str]] = frozenset({"important", "decisive"})

# The exact variables the urologist form weighted, per task. variable_weight_score
# iterates over the *ground truth* keys, so emitting all of them costs nothing
# and omitting one is silently scored as "not_used".
TASK1_VARIABLES: Final[tuple[str, ...]] = (
    "pirads", "psad", "psa", "dre", "cspca", "age", "fh", "vol", "bx", "comorbidity",
)
TASK2_VARIABLES: Final[tuple[str, ...]] = (
    "bx_gl_prim", "pirads", "bx_isup", "ct", "fh", "comorbidity",
    "psa", "bx_gl_sec", "age", "psad", "cspca",
)
VARIABLES_BY_TASK: Final[dict[int, tuple[str, ...]]] = {
    1: TASK1_VARIABLES,
    2: TASK2_VARIABLES,
}

# --------------------------------------------------------------------------- #
# Reveal vocabulary. Exactly six names; anything else counts as an extra reveal
# and is penalised by the tool-efficiency precision.
# --------------------------------------------------------------------------- #

REVEAL_SECTIONS: Final[tuple[str, ...]] = (
    "radiology_report",
    "laboratory_results",
    "psa_trend",
    "pathology_report",
    "previous_notes",
    "family_history",
)

# Variables readable from the patient card, so grounded without any reveal.
ALWAYS_AVAILABLE_VARIABLES: Final[frozenset[str]] = frozenset({"psa", "age"})

# Primary source section per variable, translated from the raw form section ids
# in section_variable_mapping.json into the flat reveal vocabulary. A variable
# whose primary sections all fall outside the six-name vocabulary is
# *ungradable* for grounding and is excluded from the score entirely -- that is
# currently only `comorbidity`, whose section cannot be declared by any
# submission.
PRIMARY_SECTIONS_BY_VARIABLE: Final[dict[str, tuple[str, ...]]] = {
    "pirads": ("radiology_report",),
    "psad": ("radiology_report",),
    "psa": ("psa_trend",),
    "dre": ("laboratory_results",),
    "cspca": ("radiology_report",),
    "age": (),
    "fh": ("family_history",),
    "vol": ("radiology_report",),
    "bx": ("pathology_report",),
    "bx_isup": ("pathology_report",),
    "bx_gl_prim": ("pathology_report",),
    "bx_gl_sec": ("pathology_report",),
    "ct": ("radiology_report",),
    "comorbidity": (),  # section_s3-comorb, outside the reveal vocabulary
}

UNGRADABLE_FOR_GROUNDING: Final[frozenset[str]] = frozenset({"comorbidity"})

__all__ = [name for name in dir() if name.isupper()]
