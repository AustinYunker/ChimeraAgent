"""Clinical guidelines, encoded.

Three published instruments, none of them fitted:

* **CAPRA-S** (Cooperberg et al., *Cancer* 2011) for Task 3 -- a post-prostatectomy
  score for biochemical recurrence built from exactly the fields the surgical
  pathology report states.
* **EAU risk stratification** for Task 2 -- ISUP grade, PSA and clinical T stage.
* A **prior-biopsy / PI-RADS partition** for Task 1.

For Tasks 1 and 2 these define *strata*, not decisions. Which label each stratum
maps to is fitted against the official ranking metric in
:mod:`chimera.cli.fit_models`, because the metric is not accuracy and the two
disagree. Keeping the partition fixed and learning only one label per leaf is what
keeps the parameter count at four or five on cohorts of 91 and 72.

Two structural findings from the released labels shaped the partitions, and both are
clinically sensible rather than incidental:

* **Task 2's prior biopsy result separates `continued_surveillance` almost
  perfectly** -- 13 of 14 negative-biopsy cases, against 1 of 58 positive ones. A
  patient with a negative biopsy continues being watched; the four-way choice only
  really arises once there is cancer to treat.
* **Task 1's decision is near-deterministic when there is no prior positive biopsy**
  (PI-RADS >= 4 goes to biopsy in 33 of 35 cases, <= 3 in none of 6) **and close to a
  coin flip when there is one** (yes-rate 0.42-0.58 across PI-RADS). The partition
  reflects that: the informative branches get their own leaves, and the ambiguous
  branch gets one leaf whose label is decided by the metric.

Pure standard library -- this runs inside the submission container.
"""

from __future__ import annotations

from typing import Any

from chimera.evidence.reports import NOT_ASSESSED, SurgicalPathology
from chimera.evidence.structured import StructuredFeatures

# --------------------------------------------------------------------------- #
# Task 3 -- CAPRA-S
# --------------------------------------------------------------------------- #

#: Maximum attainable CAPRA-S, for rescaling when a component is unreadable.
CAPRA_S_MAX = 12


def _capra_psa_points(psa: float | None) -> int | None:
    """0-3. Published bands: <=6, 6.01-10, 10.01-20, >20."""
    if psa is None or psa < 0:
        return None
    if psa <= 6:
        return 0
    if psa <= 10:
        return 1
    if psa <= 20:
        return 2
    return 3


def _capra_gleason_points(primary: int | None, secondary: int | None) -> int | None:
    """0-3. Published bands: 2-6, 3+4, 4+3, 8-10.

    Uses the *pattern pair* rather than the sum, because CAPRA-S separates 3+4 from
    4+3 -- both sum to 7 and they carry materially different risk.
    """
    if primary is None or secondary is None:
        return None
    total = primary + secondary
    if total <= 6:
        return 0
    if (primary, secondary) == (3, 4):
        return 1
    if (primary, secondary) == (4, 3):
        return 2
    return 3


def capra_s_points(
    pathology: SurgicalPathology, psa: float | None
) -> tuple[int, int] | None:
    """Raw CAPRA-S points ``(earned, assessable)``, before any rescaling.

    ``assessable`` is the sum of the maxima of the components this specimen
    actually reported, so it is :data:`CAPRA_S_MAX` only when every component
    parsed. Returns ``None`` when none of them did.

    Split out from :func:`capra_s` because the two callers want different things.
    The C-index wants the rescaled score; the rationale has to say *how much of
    the nomogram was readable*, since quoting a rescaled value as though it were
    a raw one overstates the evidence -- a specimen reporting nothing but a PSA
    over 20 rescales to a flat 12 of 12.

    Nodal status of :data:`~chimera.evidence.reports.NOT_ASSESSED` (pNx) scores zero
    points and still counts toward the denominator, matching the nomogram, which
    assigns its point only for confirmed pN1.
    """
    earned = 0
    possible = 0

    psa_points = _capra_psa_points(psa)
    if psa_points is not None:
        earned += psa_points
        possible += 3

    gleason_points = _capra_gleason_points(
        pathology.gleason_primary, pathology.gleason_secondary
    )
    if gleason_points is not None:
        earned += gleason_points
        possible += 3

    if pathology.positive_margins is not None:
        earned += 2 if pathology.positive_margins else 0
        possible += 2

    if pathology.svi is not None:
        earned += 2 if pathology.svi else 0
        possible += 2

    if pathology.epe is not None:
        earned += 1 if pathology.epe else 0
        possible += 1

    if pathology.lymph_nodes is not None:
        earned += 1 if pathology.lymph_nodes is True else 0
        possible += 1

    if possible == 0:
        return None
    return earned, possible


def capra_s(pathology: SurgicalPathology, psa: float | None) -> float | None:
    """CAPRA-S on its native 0-12 scale; higher means higher recurrence risk.

    Returns ``None`` only when nothing at all could be read. When *some* components
    are unreadable the score is computed over the rest and **rescaled to the full
    range**, so a case missing a component is not silently ranked as lower-risk than
    a fully-reported one. That matters for the C-index, which compares cases against
    each other rather than against a threshold.
    """
    points = capra_s_points(pathology, psa)
    if points is None:
        return None
    earned, possible = points
    return earned * CAPRA_S_MAX / possible


# --------------------------------------------------------------------------- #
# Task 2 -- EAU risk stratification
# --------------------------------------------------------------------------- #

#: cT2c and above. `ct_ordinal` indexes chimera.evidence.structured._CT_ORDER.
_CT_ORDINAL_T2C = 7
_CT_ORDINAL_T2B = 6


def eau_risk(features: StructuredFeatures) -> str | None:
    """``low`` / ``intermediate`` / ``high``, or ``None`` if nothing is known.

    Low needs *every* criterion satisfied; high needs *any*. An unknown field
    therefore cannot create a low-risk classification, which is the safe direction:
    under-treating a high-risk patient is the worse error clinically, and the label
    distribution agrees.
    """
    isup, psa, ct = features.bx_isup, features.psa, features.ct_ordinal
    if isup is None and psa is None and ct is None:
        return None

    if (
        (isup is not None and isup >= 4)
        or (psa is not None and psa > 20)
        or (ct is not None and ct >= _CT_ORDINAL_T2C)
    ):
        return "high"

    if (
        (isup is not None and isup <= 1)
        and (psa is not None and psa < 10)
        and (ct is not None and ct < _CT_ORDINAL_T2B)
    ):
        return "low"

    return "intermediate"


# --------------------------------------------------------------------------- #
# Strata. Each task's leaves are a small closed set; the label per leaf is fitted.
# --------------------------------------------------------------------------- #

#: Task 1 stratifies on the MRI alone. Prior-biopsy status *is* available (release
#: Version 3 dropped it from the patient card, and
#: :mod:`chimera.evidence.notes` recovers it from the prose at 99.0%), and giving
#: it its own leaf is the clinically obvious move -- imaging genuinely does not
#: discriminate in men with known cancer, 21 yes against 22 no at high PI-RADS.
#: It still loses out of fold, and not marginally:
#:
#: ===============================  ========  =======  ================
#: Task 1 stratification            accuracy  F1(yes)  pooled OOF CV
#: ===============================  ========  =======  ================
#: prior_positive as its own leaf      0.692    0.800  0.6482 +/- 0.0224
#: prior_positive split by PI-RADS         -        -  0.6679 +/- 0.0172
#: PI-RADS alone (shipped)             0.714    0.806  0.6823 +/- 0.0006
#: ===============================  ========  =======  ================
#:
#: The gap clears the repeated-CV spread that ``docs/plan.md`` sets as the bar for
#: acting at all, and the near-zero split variance is the tell: with the extra
#: leaf the fitted labels move from fold to fold, so most of what it buys is
#: noise. The more expressive six-leaf variant landing *between* the two says the
#: same thing from the other side.
#:
#: What actually changes is six patients -- prior-positive with a low-PI-RADS MRI,
#: 2 biopsied against 4 not. The extra leaf calls them all `yes`; PI-RADS alone
#: calls them all `no` and is right twice as often, which is also the defensible
#: recommendation (known cancer, unremarkable MRI, stay on surveillance).
TASK1_LEAVES: tuple[str, ...] = (
    "naive_pirads_high",
    "naive_pirads_low",
    "naive_pirads_unknown",
)

TASK2_LEAVES: tuple[str, ...] = (
    "prior_negative",
    "positive_low",
    "positive_intermediate",
    "positive_high",
    "unknown",
)

LEAVES_BY_TASK: dict[int, tuple[str, ...]] = {1: TASK1_LEAVES, 2: TASK2_LEAVES}

#: PI-RADS at or above this is a positive MRI for biopsy purposes.
PIRADS_POSITIVE = 4


def _task1_stratum(features: StructuredFeatures) -> str:
    """PI-RADS alone. Prior-biopsy status is deliberately not consulted -- see
    :data:`TASK1_LEAVES` for the measurement that decided it."""
    if features.pirads is None:
        return "naive_pirads_unknown"
    return "naive_pirads_high" if features.pirads >= PIRADS_POSITIVE else "naive_pirads_low"


def _task2_stratum(features: StructuredFeatures) -> str:
    if features.prior_biopsy == "negative":
        return "prior_negative"
    if features.prior_biopsy != "positive":
        return "unknown"
    risk = eau_risk(features)
    if risk is None:
        return "unknown"
    return f"positive_{risk}"


def stratum(task: int, features: StructuredFeatures) -> str:
    """Which guideline leaf this case falls into. Total function -- never ``None``."""
    if task == 1:
        return _task1_stratum(features)
    if task == 2:
        return _task2_stratum(features)
    raise ValueError(f"stratum is defined for tasks 1 and 2, not {task!r}")


def describe(task: int, features: StructuredFeatures) -> dict[str, Any]:
    """Human-readable trace of the guideline path, for the rationale text."""
    leaf = stratum(task, features)
    out: dict[str, Any] = {"stratum": leaf}
    if task == 2:
        out["eau_risk"] = eau_risk(features)
        # Only Task 2 branches on it, so only Task 2 may cite it as the basis.
        # Naming a fact the decision did not use would make the rationale
        # describe a path we did not take.
        if features.prior_biopsy is not None:
            out["prior_biopsy"] = features.prior_biopsy
    if features.pirads is not None:
        out["pirads"] = features.pirads
    return out
