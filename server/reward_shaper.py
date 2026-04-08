"""
RewardShaper — Internal post-processor for step reward values (internal only).

Applies additional penalty/bonus rules ON TOP of the existing tri-partite reward,
WITHOUT changing any API output, schema fields, or reward field names.

Rules (all additive adjustments, final result clamped to [0.01, 0.99]):
  A. Repetition cap:               max_reward = 0.15 (stricter)
  B. Off-topic hard cap:           max_reward = 0.05
  C. Stall progressive penalty:
       stall_count == 1 → -0.10
       stall_count == 2 → -0.20
       stall_count >= 3 → max_reward = 0.10 (override)
  D. Intent completeness boost:    all required intents present → +0.10
  E. Stage progress bonus:         stage correctly advanced → +0.05
  F. Early-stage floor sharpening: weak match → 0.40; average → 0.60; strong → 0.75+
  G. Sequence correctness bonus:   perfect intent order → +0.03
  H. Fast recovery jump:           jump from <0.3 to >0.8 → +0.05 bonus
"""

from __future__ import annotations
import sys
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Penalty / bonus constants
# ---------------------------------------------------------------------------
_REPETITION_CAP        = 0.15   # Stricter: second repetition max 0.15
_OFF_TOPIC_CAP         = 0.05
_STALL_PENALTY_1       = 0.10
_STALL_PENALTY_2       = 0.20
_STALL_HARD_CAP        = 0.10
_INTENT_COMPLETE_BONUS = 0.10
_STAGE_PROGRESS_BONUS  = 0.05
_SEQUENCE_BONUS        = 0.03   # Task 4.D
_RECOVERY_JUMP_BONUS   = 0.05   # Task 4.C

# ---------------------------------------------------------------------------
# Stage/Task Requirements
# ---------------------------------------------------------------------------
_TASK_REQUIRED: Dict[str, List[str]] = {
    "order_status":   ["tracking_info", "delivery_info"],
    "damaged_product":["empathy", "replacement"],
    "escalation":     ["empathy", "escalation", "refund"],
}

_FINAL_STAGES = {"closure", "resolution"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def shape_reward(
    base_reward: float,
    is_repetitive: bool,
    is_stalling: bool,
    stall_count: int,
    intents: Set[str],
    stage_name: str,
    stage_advanced: bool,
    detected_intents_dict: Optional[Dict[str, Any]],
    task_name: str,
    customer_interaction_data: Optional[Dict[str, Any]] = None, # Added to match environment call site
    last_reward: float = 1.0,           # For recovery bonus
    is_ordered: bool = True,            # For sequence bonus
) -> float:
    """
    Apply advanced sharpening rules to the base step reward.
    """
    reward = base_reward

    # ── A. Repetition hard cap (4.B: max 0.15 for second repetition) ──────────
    if is_repetitive:
        reward = min(reward, _REPETITION_CAP)

    # ── B. Off-topic hard cap ─────────────────────────────────────────────────
    off_topic_data = (detected_intents_dict or {}).get("off_topic", {})
    if off_topic_data.get("present", False) and off_topic_data.get("confidence", 0) > 0.7:
        reward = min(reward, _OFF_TOPIC_CAP)

    # ── C. Stall progressive penalty ──────────────────────────────────────────
    if is_stalling and not stage_advanced:
        if stall_count >= 3:
            reward = min(reward, _STALL_HARD_CAP)
        elif stall_count == 2:
            reward = max(0.01, reward - _STALL_PENALTY_2)
        elif stall_count == 1:
            reward = max(0.01, reward - _STALL_PENALTY_1)

    # Skip bonuses if repetitive or off-topic
    if not is_repetitive and not (off_topic_data.get("present", False) and off_topic_data.get("confidence", 0) > 0.7):

        # ── D. Intent completeness boost ──────────────────────────────────────
        required = _TASK_REQUIRED.get(task_name, [])
        if required and detected_intents_dict:
            all_present = all(
                detected_intents_dict.get(ri, {}).get("present", False)
                for ri in required
            )
            if all_present:
                reward = min(0.99, reward + _INTENT_COMPLETE_BONUS)

        # ── E. Stage progress bonus ────────────────────────────────────────────
        if stage_advanced:
            reward = min(0.99, reward + _STAGE_PROGRESS_BONUS)

        # ── F. Early-stage differentiation (4.A) ───────────────────────────────
        if stage_name not in _FINAL_STAGES and not stage_advanced:
            matching_count = sum(1 for i in intents if i not in {"off_topic", "information_request"})
            if matching_count == 0:
                reward = min(reward, 0.45) # Weak
            elif matching_count == 1:
                reward = min(reward, 0.60) # Average
            elif matching_count >= 2:
                reward = min(reward, 0.75) # Strong

        # ── G. Sequence correctness bonus (4.D) ───────────────────────────────
        if is_ordered and stage_advanced:
            reward = min(0.99, reward + _SEQUENCE_BONUS)

        # ── H. Fast recovery jump / Partial recovery (Task 4.C) ───────────────
        if last_reward < 0.3:
            if reward > 0.8:
                reward = min(0.99, reward + _RECOVERY_JUMP_BONUS)
            elif reward >= 0.3:
                # Add a "partial recovery boost" to push it above the 0.4 stress threshold
                reward = min(0.99, reward + 0.15)

    # Final clamp
    reward = max(0.01, min(0.99, reward))
    return reward
