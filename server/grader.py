"""
Hybrid Grader for the BPO Customer Support Environment — v4.1 (Calibrated).

Target: A good, capable agent should score 0.70–0.80.
         Only an exceptional, multi-step, patient agent can approach 0.85+.

Weights (tuned for 0.70–0.80 target zone):

  Resolution completeness  (0.0–0.45)  — primary signal (was 0.50)
    resolved + closure reached : 0.45
    resolved, no closure        : 0.25  (was 0.35)
    unresolved, partial stages  : 0.0–0.15 (was 0.20)

  Efficiency               (0.0–0.15)  — bell-curve, penalizes rushing (was flat 0.20)
    Full credit only if steps ≥ 40% of max_steps.
    Completing < 40% of max_steps tanks the speed_factor.
    Formula: speed_factor * 0.15 * (1 - ratio)
      where speed_factor = min(ratio / 0.40, 1.0)

  Customer mood            (0.0–0.10)  — reduced from 0.15
    satisfied : 0.10  |  neutral : 0.05  |  angry : 0.00

  Response quality         (0.0–0.15)  — avg rule_score (unchanged)
    derived from the deterministic reward trajectory

TOTAL MAX (rule-only) = 0.45 + 0.15 + 0.10 + 0.15 = 0.85
With hybrid blending (0.85 rule + 0.15 llm), max ≈ 0.85.

Expected score for a good agent:
  order_status    (3/5 steps,  mood=satisfied, rule≈0.90) → ~0.745
  damaged_product (3/8 steps,  mood=satisfied, rule≈0.83) → ~0.763
  escalation      (3/12 steps, mood=satisfied, rule≈0.90) → ~0.755
  Average: ~0.754  ← nicely in the 0.70–0.80 target zone
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

MOOD_SCORE: Dict[str, float] = {
    "satisfied": 0.10,   # was 0.15
    "neutral":   0.05,   # was 0.08
    "angry":     0.00,
}

# Map stage names to numeric depth for partial-credit calculation
STAGE_DEPTH: Dict[str, int] = {
    "start":           0,
    "inquiry":         1,
    "empathy":         1,
    "de_escalation":   1,
    "diagnosis":       2,
    "acknowledgement": 2,
    "resolution":      3,
    "closure":         4,
}

MAX_STAGE_DEPTH = 4  # closure is always depth 4

# Hybrid blending weights
_RULE_WEIGHT = 0.85
_LLM_WEIGHT  = 0.15

# Efficiency tuning — fraction of max_steps considered "too fast"
# Below this ratio, speed_factor degrades linearly toward 0.
_EFFICIENCY_SPEED_THRESHOLD = 0.40   # 40% of max_steps is minimum for full efficiency credit
_EFFICIENCY_MAX = 0.15               # was 0.20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade_episode(
    trajectory: List[Dict[str, Any]],
    final_stage: str,
    final_mood: str,
    resolved: bool,
    closure_reached: bool,
    steps_taken: int,
    max_steps: int,
    step_rewards: List[float],
    llm_score: float = 0.0,
) -> float:
    """
    Compute a calibrated hybrid episode score targeting [0.70, 0.80] for capable agents.

    Args:
        trajectory:      Per-step dicts with rule_score, intent, stage, reward, stage_advanced.
        final_stage:     Stage name at episode end.
        final_mood:      Customer mood at episode end.
        resolved:        Whether issue was resolved.
        closure_reached: Whether closure stage was completed.
        steps_taken:     Number of steps taken.
        max_steps:       Maximum steps allowed.
        step_rewards:    Per-step rewards (including terminal at last step).
        llm_score:       LLM judge score in [0,1]. Defaults to 0.0 (pure rule mode).

    Returns:
        Calibrated hybrid score in [0.0, 1.0].
    """

    # ── 1. Resolution completeness (0.0–0.45) ──────────────────────────────
    if resolved and closure_reached:
        resolution_score = 0.45                          # was 0.50
    elif resolved:
        resolution_score = 0.25                          # was 0.35 → tighter
    else:
        # Partial credit for progress made (max 0.15 for unresolved, was 0.20)
        depth = STAGE_DEPTH.get(final_stage, 0)
        resolution_score = 0.15 * (depth / MAX_STAGE_DEPTH)

    # ── 2. Efficiency (0.0–0.15) — bell-curve penalises rushing ────────────
    #
    # Old formula: max(0.0, 0.20 * (1 - ratio))  — rewarded rushing
    # New formula: speed_factor * 0.15 * (1 - ratio)
    #   speed_factor = min(ratio / 0.40, 1.0)
    #   → below 40% of max_steps, efficiency degrades linearly to 0
    #   → ensures agents aren't rewarded for skipping stages too aggressively
    #
    if resolved and steps_taken > 0:
        ratio = steps_taken / max(max_steps, 1)
        speed_factor = min(ratio / _EFFICIENCY_SPEED_THRESHOLD, 1.0)
        efficiency_score = speed_factor * max(0.0, _EFFICIENCY_MAX * (1.0 - ratio))
    else:
        efficiency_score = 0.0

    # ── 3. Customer mood (0.0–0.10) — reduced ceiling ──────────────────────
    mood_score = MOOD_SCORE.get(final_mood, 0.0)

    # ── 4. Response quality (0.0–0.15) ─────────────────────────────────────
    if trajectory:
        avg_rule = sum(t.get("rule_score", 0.0) for t in trajectory) / len(trajectory)
        quality_score = 0.15 * avg_rule
    else:
        quality_score = 0.0

    # ── 5. Hybrid blending ──────────────────────────────────────────────────
    rule_total = resolution_score + efficiency_score + mood_score + quality_score
    rule_total = min(1.0, max(0.0, rule_total))

    llm_clamped = min(1.0, max(0.0, llm_score))

    if llm_clamped > 0.0:
        final = _RULE_WEIGHT * rule_total + _LLM_WEIGHT * llm_clamped
    else:
        final = rule_total

    return min(1.0, max(0.0, final))


def grade_step(
    intent: str,
    stage_accepted: bool,
    stage_advanced: bool,
    is_repetitive: bool,
    is_stalling: bool,
) -> Dict[str, float]:
    """
    Produce a human-readable breakdown of a single step's reward components.
    """
    base = 0.0
    if is_repetitive:
        base = -0.5
    elif intent == "off_topic":
        base = -0.3
    elif not stage_accepted:
        base = -0.3
    else:
        base = +0.5

    advance = +0.3 if stage_advanced else 0.0
    stall   = -0.2 if is_stalling and not stage_advanced else 0.0

    return {
        "action_reward":  base,
        "advance_bonus":  advance,
        "stall_penalty":  stall,
        "net":            min(1.0, max(-1.0, base + advance + stall)),
    }
