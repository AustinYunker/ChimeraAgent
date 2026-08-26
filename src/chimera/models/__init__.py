"""Decision models.

The shape shared by Tasks 1 and 2 is a **guideline-stratified, metric-fitted rule**:
clinical knowledge decides how cases are partitioned, and the label assigned to each
partition is fitted to maximise the official ranking score.

That split matters for two reasons. With 91 and 72 labeled cases, a model that learns
its own partitions has far too much freedom; a model that learns one label per
guideline stratum has four or five parameters. And the metric is not accuracy --
Task 1 scores ``(mean_case_score + F1_yes) / 2``, so the label that maximises correct
answers at a leaf is not always the label that maximises the score.

Task 3 needs no fitting at all: CAPRA-S is a published nomogram and only the ordering
of its output affects Harrell's C-index.
"""

from chimera.models.guidelines import (
    LEAVES_BY_TASK,
    capra_s,
    capra_s_points,
    eau_risk,
    stratum,
)

__all__ = ["LEAVES_BY_TASK", "capra_s", "capra_s_points", "eau_risk", "stratum"]
