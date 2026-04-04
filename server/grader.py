"""
Deterministic Grader for the BPO Customer Support Environment.

Produces a reproducible episode score in [0.0, 1.0] based on:
  - Resolution quality  (0.0–0.40)
  - Efficiency          (0.0–0.20)  fewer steps is better
  - Customer mood       (0.0–0.20)  final mood of the customer
  - Response quality    (0.0–0.20)  average rule-based score across turns

This grader is called ONCE at episode end and never returns a constant;
every dimension is computed from the actual episode trajectory.
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Mood weights
# ---------------------------------------------------------------------------

MOOD_SCORE: Dict[str, float] = {
    "satisfied": 0.20,
    "neutral":   0.10,
    "angry":     0.00,
}

# ---------------------------------------------------------------------------
# Stage completion weights (fraction of full stage list reached)
# ---------------------------------------------------------------------------

STAGE_ORDER: Dict[str, int] = {
    # order_status
    "start":           0,
    "inquiry":         1,
    "empathy":         1,   # shared with damaged_product / escalation
    "diagnosis":       2,
    "de_escalation":   1,
    "acknowledgement": 2,
    "resolution":      3,
    "closure":         4,
}

MAX_STAGE_INDEX = 4  # closure is always index 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade_episode(
    trajectory: List[Dict[str, Any]],
    final_stage: str,
    final_mood: str,
    resolved: bool,
    steps_taken: int,
    max_steps: int,
) -> float:
    """
    Compute a deterministic episode score in [0.0, 1.0].

    Args:
        trajectory:   List of per-step dicts with keys:
                        rule_score (float), intent (str), stage (str)
        final_stage:  The stage name at episode end.
        final_mood:   Customer mood at episode end ("angry"|"neutral"|"satisfied").
        resolved:     Whether the episode ended with issue resolved.
        steps_taken:  Number of steps taken.
        max_steps:    Maximum steps allowed.

    Returns:
        Composite score in [0.0, 1.0].
    """
    # --- 1. Resolution score (0.0–0.40) ---
    if resolved:
        resolution_score = 0.40
    else:
        # Partial credit for reaching later stages even if not fully resolved
        stage_idx = STAGE_ORDER.get(final_stage, 0)
        resolution_score = 0.40 * (stage_idx / MAX_STAGE_INDEX) * 0.5  # max 0.20 partial

    # --- 2. Efficiency score (0.0–0.20) ---
    if steps_taken <= 0:
        efficiency_score = 0.0
    else:
        # Full score if resolved in ≤ half max_steps; degrades linearly
        ratio = steps_taken / max(max_steps, 1)
        efficiency_score = max(0.0, 0.20 * (1.0 - ratio))

    # --- 3. Mood score (0.0–0.20) ---
    mood_score = MOOD_SCORE.get(final_mood, 0.0)

    # --- 4. Response quality score (0.0–0.20) ---
    if trajectory:
        avg_rule = sum(t.get("rule_score", 0.0) for t in trajectory) / len(trajectory)
        quality_score = 0.20 * avg_rule
    else:
        quality_score = 0.0

    raw = resolution_score + efficiency_score + mood_score + quality_score
    return min(1.0, max(0.0, raw))


def grade_step(
    intent: str,
    rule_score: float,
    stage_advanced: bool,
) -> Dict[str, float]:
    """
    Produce a human-readable breakdown of a single step's reward components.
    Useful for logging / debugging.
    """
    return {
        "base":             0.50,
        "stage_advance":    0.30 if stage_advanced else 0.0,
        "rule_quality":     rule_score - 0.50,   # delta from base
        "net":              min(1.0, max(0.0, 0.50 + (0.30 if stage_advanced else 0.0) + (rule_score - 0.50))),
    }
