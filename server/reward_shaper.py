"""
RewardShaper — Internal post-processor for step reward values (internal only).

Applies additional penalty/bonus rules ON TOP of the existing tri-partite reward,
WITHOUT changing any API output, schema fields, or reward field names.

This module is called inside _compute_step_reward() in bpo_env_environment.py
AFTER the normal computation. It returns only a modified float — the same
`step_reward` variable — so the observation schema is completely unchanged.

Rules (all additive adjustments, final result clamped to [0.01, 0.99]):
  A. Repetition cap:               max_reward = 0.20
  B. Off-topic hard cap:           max_reward = 0.05
  C. Stall progressive penalty:
       stall_count == 1 → -0.10
       stall_count == 2 → -0.20
       stall_count >= 3 → max_reward = 0.10 (override)
  D. Intent completeness boost:    all required intents present → +0.10
  E. Stage progress bonus:         stage correctly advanced → +0.05
  F. Early-stage floor sharpening: at non-closure stage with weak match → cap 0.40

Debug log lines are written to stderr ONLY (never in API response).
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Penalty / bonus constants
# ---------------------------------------------------------------------------

_REPETITION_CAP       = 0.20
_OFF_TOPIC_CAP        = 0.05
_STALL_PENALTY_1      = 0.10
_STALL_PENALTY_2      = 0.20
_STALL_HARD_CAP       = 0.10   # stall_count >= 3
_INTENT_COMPLETE_BONUS= 0.10
_STAGE_PROGRESS_BONUS = 0.05
_EARLY_STAGE_CAP      = 0.40   # weak match at early stage

# Enable/disable debug logging (stderr only — never in API output)
# _DEBUG_SHAPER = False (REMOVED to ensure zero output change)



# ---------------------------------------------------------------------------
# TASK_REQUIREMENTS mirror (to avoid circular import)
# We redeclare a minimal version here; the full version lives in bpo_env_environment.py
# ---------------------------------------------------------------------------
_TASK_REQUIRED: Dict[str, List[str]] = {
    "order_status":   ["tracking_info", "delivery_info"],
    "damaged_product":["empathy", "replacement"],
    "escalation":     ["empathy", "escalation", "refund"],
}

# Closure/final-stage names (no early-stage cap applies here)
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
) -> float:
    """
    Apply shaping rules to the base step reward.

    Args:
        base_reward:           The reward computed by _compute_step_reward before shaping.
        is_repetitive:         True if repetition was detected.
        is_stalling:           True if stall threshold exceeded.
        stall_count:           Number of consecutive non-advancing steps.
        intents:               Set of bridged intent labels for this step.
        stage_name:            Current stage name.
        stage_advanced:        Whether the stage advanced this step.
        detected_intents_dict: Full confidence-aware intent dict from intents.py.
        task_name:             Current task name.

    Returns:
        Shaped reward (float), clamped to [0.01, 0.99].
    """
    reward = base_reward
    log_parts: List[str] = [f"base={reward:.3f}"]

    # ── A. Repetition hard cap ────────────────────────────────────────────────
    if is_repetitive:
        reward = min(reward, _REPETITION_CAP)
        log_parts.append(f"repetition_cap→{_REPETITION_CAP}")


    # ── B. Off-topic hard cap ─────────────────────────────────────────────────
    off_topic_data = (detected_intents_dict or {}).get("off_topic", {})
    if off_topic_data.get("present", False) and off_topic_data.get("confidence", 0) > 0.7:
        reward = min(reward, _OFF_TOPIC_CAP)
        log_parts.append(f"off_topic_cap→{_OFF_TOPIC_CAP}")


    # ── C. Stall progressive penalty ──────────────────────────────────────────
    if is_stalling and not stage_advanced:
        if stall_count >= 3:
            reward = min(reward, _STALL_HARD_CAP)
            log_parts.append(f"stall_hard_cap[{stall_count}]→{_STALL_HARD_CAP}")

        elif stall_count == 2:
            reward = max(0.01, reward - _STALL_PENALTY_2)
            log_parts.append(f"stall_pen_2:-{_STALL_PENALTY_2}")

        elif stall_count == 1:
            reward = max(0.01, reward - _STALL_PENALTY_1)
            log_parts.append(f"stall_pen_1:-{_STALL_PENALTY_1}")


    # Skip bonuses if repetitive or off-topic (don't reward bad behavior)
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
                log_parts.append(f"intent_complete:+{_INTENT_COMPLETE_BONUS}")


        # ── E. Stage progress bonus ────────────────────────────────────────────
        if stage_advanced:
            reward = min(0.99, reward + _STAGE_PROGRESS_BONUS)
            log_parts.append(f"stage_progress:+{_STAGE_PROGRESS_BONUS}")


        # ── F. Early-stage floor sharpening ───────────────────────────────────
        # Weak match at early stages should be ~0.4, not 0.6
        if stage_name not in _FINAL_STAGES and not stage_advanced:
            matching_count = sum(
                1 for i in intents
                if i not in {"off_topic", "information_request"}
            )
            if matching_count < 2 and reward > _EARLY_STAGE_CAP:
                reward = _EARLY_STAGE_CAP
                log_parts.append(f"early_stage_cap→{_EARLY_STAGE_CAP}")


    # ── Final clamp ───────────────────────────────────────────────────────────
    reward = max(0.01, min(0.99, reward))


    return reward
