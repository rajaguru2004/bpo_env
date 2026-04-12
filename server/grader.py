"""
Pure Rule-Based Grader for the BPO Customer Support Environment — v9 (Intent-Coverage Aware).

Grader score is INDEPENDENT of per-step rewards (step_reward / rule_score).
  - step_reward  = per-step RL signal (tripartite + confidence adjustments)
  - rule_score   = per-step pure tripartite score (intent + completeness + sequence)
  - grader_score = episode-level quality evaluation (this module)

Components (all independent, no LLM blending):

  Resolution completeness  (0.0–0.45)  — primary signal
    resolved + closure reached : 0.45
    resolved, no closure        : 0.25
    unresolved, partial stages  : 0.0–0.15

  Efficiency               (0.0–0.15)  — bell-curve, penalizes rushing
    Full credit only if steps ≥ 40% of max_steps.

  Customer mood            (0.0–0.10)
    satisfied : 0.10  |  neutral : 0.05  |  angry : 0.00

  Response quality         (0.0–0.15)  — avg completeness_score across trajectory

  Intent coverage          (0.0–0.15)  — episode-level required-intent detection
    Fraction of required intents confidently detected (≥0.5) at least once.
    e.g. order_status needs tracking_info + delivery_info for full credit.

TOTAL MAX = 0.45 + 0.15 + 0.10 + 0.15 + 0.15 = 1.00
Result is clamped to [0.0, 1.0].
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

MOOD_SCORE: Dict[str, float] = {
    "satisfied": 0.10,
    "neutral":   0.05,
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

# Efficiency tuning — fraction of max_steps considered "too fast"
# Below this ratio, speed_factor degrades linearly toward 0.
_EFFICIENCY_SPEED_THRESHOLD = 0.40   # 40% of max_steps is minimum for full efficiency credit
_EFFICIENCY_MAX = 0.15


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
    llm_score: float = 0.0,  # kept for API compatibility, ignored
    required_intents: List[str] = None,
) -> float:
    """
    Compute a calibrated rule-based episode score targeting [0.70, 0.80] for capable agents.

    Args:
        trajectory:      Per-step dicts with rule_score, intent, stage, reward, stage_advanced.
        final_stage:     Stage name at episode end.
        final_mood:      Customer mood at episode end.
        resolved:        Whether issue was resolved.
        closure_reached: Whether closure stage was completed.
        steps_taken:     Number of steps taken.
        max_steps:       Maximum steps allowed.
        step_rewards:    Per-step rewards (for compatibility, not used in this grader).
        llm_score:       Ignored. Kept for backwards-compatibility only.
        required_intents: List of required intents for the task.

    Returns:
        Calibrated score in [0.0, 1.0].
    """

    # ── 1. Resolution completeness (0.0–0.45) ──────────────────────────────
    if resolved and closure_reached:
        resolution_score = 0.45
    elif resolved:
        resolution_score = 0.25
    else:
        # Partial credit for progress made (max 0.15 for unresolved)
        depth = STAGE_DEPTH.get(final_stage, 0)
        resolution_score = 0.15 * (depth / MAX_STAGE_DEPTH)

    # ── 2. Efficiency (0.0–0.15) — bell-curve penalises rushing ────────────
    #
    # speed_factor rises linearly from 0 to 1 as ratio goes from 0 to 0.40.
    # Above 0.40, speed_factor = 1. Efficiency then decays as ratio → 1.
    # This prevents rewards for skipping stages too aggressively.
    #
    if resolved and steps_taken > 0:
        ratio = steps_taken / max(max_steps, 1)
        speed_factor = min(ratio / _EFFICIENCY_SPEED_THRESHOLD, 1.0)
        efficiency_score = speed_factor * max(0.0, _EFFICIENCY_MAX * (1.0 - ratio))
    else:
        efficiency_score = 0.0

    # ── 3. Customer mood (0.0–0.10) ─────────────────────────────────────────
    mood_score = MOOD_SCORE.get(final_mood, 0.0)

    # ── 4. Response quality (0.0–0.15) ──────────────────────────────────────────
    # Use completeness_score from trajectory (v6). Fall back to rule_score for
    # backward compatibility with trajectories produced by older environment versions.
    if trajectory:
        if "completeness_score" in trajectory[0]:
            avg_quality = sum(
                t.get("completeness_score", 0.0) for t in trajectory
            ) / len(trajectory)
        else:
            # Backward-compat: use rule_score
            avg_quality = sum(
                t.get("rule_score", 0.0) for t in trajectory
            ) / len(trajectory)
        quality_score = 0.15 * avg_quality
    else:
        quality_score = 0.0

    # ── 5. Intent coverage bonus (0.0–0.15) ─────────────────────────────────
    # Rewards episode-level coverage of required intents using the
    # confidence-aware detected_intents stored in each trajectory step.
    # Distinct from step rewards: measures whether the WHOLE episode
    # addressed every required signal at least once (confidence >= 0.5).
    if required_intents:
        score_count = 0
        for ri in required_intents:
            for t in trajectory:
                raw = t.get("detected_intents", {}).get(ri)
                if raw is None:
                    continue
                if isinstance(raw, dict):
                    if raw.get("present", False) and raw.get("confidence", 0.0) >= 0.5:
                        score_count += 1
                        break
                elif raw:  # legacy bool
                    score_count += 1
                    break

        intent_coverage = score_count / len(required_intents)
        intent_bonus = 0.15 * intent_coverage
    else:
        intent_bonus = 0.0

    # ── 6. Total (pure rule-based, no LLM blending) ──────────────────────────
    # Components:  resolution(0-0.45) + efficiency(0-0.15) + mood(0-0.10)
    #              + quality(0-0.15)  + intent_bonus(0-0.15)  = MAX 1.00
    total = resolution_score + efficiency_score + mood_score + quality_score + intent_bonus

    return round(max(0.01, min(0.99, total)), 2)



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
        base = -0.15
    elif intent == "off_topic":
        base = -0.10
    elif not stage_accepted:
        base = -0.10
    else:
        base = +0.15

    stall = -0.05 if is_stalling and not stage_advanced else 0.0

    return {
        "action_reward": base,
        "stall_penalty": stall,
        "net":           min(0.15, max(-0.15, base + stall)),
    }
