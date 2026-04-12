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

# Aliases for the hackathon validator (task ids vs names)
TASK_CONFIGS["task_easy"]   = TASK_CONFIGS["order_status"]
TASK_CONFIGS["task_medium"] = TASK_CONFIGS["damaged_product"]
TASK_CONFIGS["task_hard"]   = TASK_CONFIGS["escalation"]


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
    # Round to 2 decimal places for consistent output
    return round(max(0.01, min(0.99, total)), 2)


# ---------------------------------------------------------------------------
# Per-Task Grader Functions — Phase 2 validator discovers these individually
# Each function grades a single agent *response* string against optional state.
# Signature: (response: str, state: dict) -> float in [0.0, 1.0]
# ---------------------------------------------------------------------------

def grade_order_status(response: str, state: dict = None) -> float:
    """
    Grade a single agent response for the order_status task.

    Scoring (no free base — blank responses score 0.01):
      +0.30  tracking number / tracking ID provided
      +0.25  shipment status clearly communicated
      +0.20  expected / estimated delivery date mentioned
      +0.15  professional greeting or closure phrase
      +0.10  case / reference number given
    Total max = 1.00  (clamped to [0.01, 0.99])
    """
    if state is None:
        state = {}
    # Blank / very short response gets minimum score
    if not response or len(response.strip().split()) < 4:
        return 0.01

    text = response.lower()
    score = 0.0  # No free base — must actually say something useful

    if any(w in text for w in ["tracking number", "tracking id", "trk", "tracking #",
                                "shipment tracking"]):
        score += 0.30
    if any(w in text for w in ["shipped", "delivered", "in transit", "on its way",
                                "dispatched", "out for delivery", "processed",
                                "order status"]):
        score += 0.25
    if any(w in text for w in ["expected delivery", "estimated delivery", "delivery date",
                                "arrive", "arrival", "deliver by",
                                "april", "march", "may", "days"]):
        score += 0.20
    if any(w in text for w in ["hello", "hi", "thank you for reaching", "happy to help",
                                "glad to help", "anything else", "have a great",
                                "you're welcome", "my pleasure"]):
        score += 0.15
    if any(w in text for w in ["case number", "reference number", "ticket number",
                                "case id", "ref #"]):
        score += 0.10

    return round(max(0.01, min(0.99, score)), 2)


def grade_damaged_product(response: str, state: dict = None) -> float:
    """
    Grade a single agent response for the damaged_product task.

    Scoring (no free base — blank responses score 0.01):
      +0.25  apology / regret expressed
      +0.20  empathy / understanding shown
      +0.25  replacement or refund clearly offered
      +0.15  timeline / ETA given for resolution
      +0.15  case / reference number provided
    Total max = 1.00  (clamped to [0.01, 0.99])
    """
    if state is None:
        state = {}
    if not response or len(response.strip().split()) < 4:
        return 0.01

    text = response.lower()
    score = 0.0  # No free base

    if any(w in text for w in ["sorry", "apologize", "apologies", "regret",
                                "sincerely apologize", "deeply apologize",
                                "i'm sorry", "i am sorry"]):
        score += 0.25
    if any(w in text for w in ["understand", "hear you", "concern", "frustrat",
                                "inconvenience", "appreciate your patience",
                                "empathize", "i can imagine"]):
        score += 0.20
    if any(w in text for w in ["replacement", "replace", "refund", "new unit",
                                "send a new", "ship a new", "arrange",
                                "new product", "exchange"]):
        score += 0.25
    if any(w in text for w in ["days", "hours", "business days", "24", "48",
                                "3-5", "week", "timeline", "within"]):
        score += 0.15
    if any(w in text for w in ["case number", "reference number", "ticket number",
                                "case id", "ref", "confirmation"]):
        score += 0.15

    return round(max(0.01, min(0.99, score)), 2)


def grade_escalation(response: str, state: dict = None) -> float:
    """
    Grade a single agent response for the escalation (hard) task.

    This grader is more demanding — frontier models must work for a high score:
      +0.20  genuine apology and empathy demonstrated
      +0.20  refund or compensation explicitly committed
      +0.20  manager / supervisor escalation offered
             (only full credit at step >= 2; partial 0.10 at step 1 to require de-escalation first)
      +0.15  concrete resolution timeline stated
      +0.15  case / reference number provided
      +0.10  de-escalation language beyond generic apology
    Total max = 1.00  (clamped to [0.01, 0.99])

    Args:
        response: The agent's response string.
        state:    Dict with optional keys:
                    "step" (int)  — current step number (1-based)
                    "stage" (str) — current conversation stage
    """
    if state is None:
        state = {}
    if not response or len(response.strip().split()) < 4:
        return 0.01

    text = response.lower()
    score = 0.0
    step = int(state.get("step", 1))

    # 1. Apology + empathy (must be more than a token sorry)
    apology_match = any(w in text for w in ["sorry", "apologize", "sincerely",
                                             "apologies", "deeply sorry",
                                             "i understand", "i hear you"])
    if apology_match:
        score += 0.20

    # 2. Refund / compensation commitment
    if any(w in text for w in ["refund", "full refund", "compensation", "credit",
                                "process your refund", "issue a refund",
                                "reimburse"]):
        score += 0.20

    # 3. Manager / escalation — ONLY full credit after step 1
    #    At step 1 the agent should first de-escalate, not immediately escalate.
    escalation_match = any(w in text for w in ["manager", "supervisor", "escalate",
                                                "senior", "specialist", "team lead",
                                                "transfer", "connect you with"])
    if escalation_match:
        if step >= 2:
            score += 0.20   # Full credit — de-escalation happened first
        else:
            score += 0.10   # Partial — jumped straight to escalation at step 1

    # 4. Timeline / commitment
    if any(w in text for w in ["processed", "within", "days", "hours",
                                "48", "24", "3-5", "business days",
                                "by tomorrow", "immediately"]):
        score += 0.15

    # 5. Case / reference number (shows procedural closure)
    if any(w in text for w in ["case number", "reference number", "ticket number",
                                "case id", "confirmation number", "ref #"]):
        score += 0.15

    # 6. De-escalation beyond generic sorry (calming, taking ownership)
    deescalation_extra = any(w in text for w in [
        "i take full responsibility", "i take ownership",
        "i assure you", "rest assured", "i personally",
        "i will make sure", "i will ensure", "i will personally",
        "you have my word", "i promise",
    ])
    if deescalation_extra:
        score += 0.10

    return round(max(0.01, min(0.99, score)), 2)


# ---------------------------------------------------------------------------
# TASKS Registry — central map of task name → config + grader
# The validator uses this to confirm all 3 tasks are enabled and have graders.
# ---------------------------------------------------------------------------

TASKS: Dict[str, Any] = {
    "order_status": {
        "config": TASK_CONFIGS["order_status"],
        "grader": grade_order_status,
        "grader_name": "grade_order_status",
        "enabled": True,
    },
    "damaged_product": {
        "config": TASK_CONFIGS["damaged_product"],
        "grader": grade_damaged_product,
        "grader_name": "grade_damaged_product",
        "enabled": True,
    },
    "escalation": {
        "config": TASK_CONFIGS["escalation"],
        "grader": grade_escalation,
        "grader_name": "grade_escalation",
        "enabled": True,
    },
}

# Aliases for the hackathon validator (task ids vs names)
TASKS["task_easy"]   = TASKS["order_status"]
TASKS["task_medium"] = TASKS["damaged_product"]
TASKS["task_hard"]   = TASKS["escalation"]

