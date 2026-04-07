"""
tasks.py — BPO Customer Support Environment
============================================
OpenEnv-compliant task registry and grader for the hackathon validator.

Defines TASK_CONFIGS for the 3 tasks and provides a top-level
grade_episode(task_name, trajectory) function that the OpenEnv
validator discovers to confirm graders are present.

This module is intentionally self-contained (no local imports)
so the validator can import it in isolation without any sys.path tricks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Task Configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskConfig:
    """Immutable configuration for a single task."""
    name: str
    difficulty: str
    description: str
    max_steps: int


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "order_status": TaskConfig(
        name="order_status",
        difficulty="easy",
        max_steps=5,
        description="Customer wants to know the status of their order.",
    ),
    "damaged_product": TaskConfig(
        name="damaged_product",
        difficulty="medium",
        max_steps=8,
        description="Customer received a damaged product and wants a replacement or refund.",
    ),
    "escalation": TaskConfig(
        name="escalation",
        difficulty="hard",
        max_steps=12,
        description="Angry customer demanding a full refund and to speak with a manager.",
    ),
}

DEFAULT_TASK = "order_status"


# ---------------------------------------------------------------------------
# Grader helpers (inlined — no local imports so the validator can import this
# file in isolation without needing any sys.path manipulation)
# ---------------------------------------------------------------------------

# Map stage names to numeric depth for partial-credit calculation
_STAGE_DEPTH: Dict[str, int] = {
    "start":           0,
    "inquiry":         1,
    "empathy":         1,
    "de_escalation":   1,
    "diagnosis":       2,
    "acknowledgement": 2,
    "resolution":      3,
    "closure":         4,
}

_MAX_STAGE_DEPTH = 4

# Efficiency tuning
_EFFICIENCY_SPEED_THRESHOLD = 0.40
_EFFICIENCY_MAX = 0.15

# Mood score mapping
_MOOD_SCORE: Dict[str, float] = {
    "satisfied": 0.10,
    "neutral":   0.05,
    "angry":     0.00,
}


# ---------------------------------------------------------------------------
# Public API — the validator calls grade_episode(task_name, trajectory)
# ---------------------------------------------------------------------------

def grade_episode(task_name: str, trajectory: Any) -> float:
    """
    Score an episode trajectory on a 0.0 – 1.0 scale.

    This is the primary grader entry point used by the OpenEnv validator.
    Accepts both list (per-step dicts) and dict (episode summary) formats.

    Parameters
    ----------
    task_name : str
        One of: "order_status", "damaged_product", "escalation"
    trajectory : list or dict
        Either:
        - A list of per-step observation dicts produced by the environment
        - A summary dict with keys: resolved, closure_reached, steps_taken,
          max_steps, final_stage, final_mood, and optionally "trajectory"
          (list of per-step dicts for quality scoring)

    Returns
    -------
    float in [0.01, 0.99]
        Never exactly 0.0 or 1.0 to satisfy strict validator bounds.
    """
    task_cfg = TASK_CONFIGS.get(task_name, TASK_CONFIGS[DEFAULT_TASK])

    # ── Normalise input format ────────────────────────────────────────────────
    if isinstance(trajectory, list):
        steps = trajectory
        if not steps:
            return 0.01
        final_step = steps[-1]
        resolved        = bool(final_step.get("is_resolved", False))
        closure_reached = bool(final_step.get("success", False))
        steps_taken     = len(steps)
        max_steps       = int(final_step.get("max_steps", task_cfg.max_steps))
        final_stage     = str(final_step.get("conversation_stage", "start"))
        final_mood      = str(final_step.get("customer_mood", "neutral"))
        quality_scores  = [
            float(s.get("completeness_score", s.get("rule_score", 0.0)))
            for s in steps
        ]
    elif isinstance(trajectory, dict):
        steps           = trajectory.get("trajectory", trajectory.get("steps", []))
        resolved        = bool(trajectory.get("resolved", False))
        closure_reached = bool(trajectory.get("closure_reached", False))
        steps_taken     = int(trajectory.get("steps_taken", len(steps)))
        max_steps       = int(trajectory.get("max_steps", task_cfg.max_steps))
        final_stage     = str(trajectory.get("final_stage", "start"))
        final_mood      = str(trajectory.get("final_mood", "neutral"))
        quality_scores  = [
            float(s.get("completeness_score", s.get("rule_score", 0.0)))
            for s in steps
        ]
    else:
        return 0.01

    # ── 1. Resolution completeness (0.0 – 0.45) ──────────────────────────────
    if resolved and closure_reached:
        resolution_score = 0.45
    elif resolved:
        resolution_score = 0.25
    else:
        depth = _STAGE_DEPTH.get(final_stage, 0)
        resolution_score = 0.15 * (depth / _MAX_STAGE_DEPTH)

    # ── 2. Efficiency (0.0 – 0.15) ───────────────────────────────────────────
    if resolved and steps_taken > 0:
        ratio        = steps_taken / max(max_steps, 1)
        speed_factor = min(ratio / _EFFICIENCY_SPEED_THRESHOLD, 1.0)
        efficiency_score = speed_factor * max(0.0, _EFFICIENCY_MAX * (1.0 - ratio))
    else:
        efficiency_score = 0.0

    # ── 3. Customer mood (0.0 – 0.10) ────────────────────────────────────────
    mood_score = _MOOD_SCORE.get(final_mood, 0.05)

    # ── 4. Response quality (0.0 – 0.15) — avg completeness across steps ──────
    if quality_scores:
        avg_quality   = sum(quality_scores) / len(quality_scores)
        quality_score = 0.15 * avg_quality
    else:
        quality_score = 0.0

    # ── Total (max = 0.45 + 0.15 + 0.10 + 0.15 = 0.85) ──────────────────────
    total = resolution_score + efficiency_score + mood_score + quality_score

    # Clamp to strict (0.01, 0.99) — validator rejects exactly 0.0 and 1.0
    return max(0.01, min(0.99, total))
