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
_EFFICIENCY_FLOOR = 0.05   # Minimum efficiency credit for any resolved episode

# Mood score mapping
_MOOD_SCORE: Dict[str, float] = {
    "satisfied": 0.10,
    "neutral":   0.05,
    "angry":     0.00,
}

# Required intents per task for intent-coverage bonus
_TASK_REQUIRED_INTENTS: Dict[str, List[str]] = {
    "order_status":    ["tracking_info", "delivery_info"],
    "damaged_product": ["empathy", "replacement"],
    "escalation":      ["empathy", "escalation", "refund"],
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
    # Bell-curve: rises to 1.0 at 40% of max_steps, then decays.
    # Floor of _EFFICIENCY_FLOOR ensures resolved episodes always get some credit
    # regardless of how many steps were used (prevents punishing thorough agents).
    if resolved and steps_taken > 0:
        ratio        = steps_taken / max(max_steps, 1)
        speed_factor = min(ratio / _EFFICIENCY_SPEED_THRESHOLD, 1.0)
        raw_eff      = speed_factor * max(0.0, _EFFICIENCY_MAX * (1.0 - ratio))
        efficiency_score = max(raw_eff, _EFFICIENCY_FLOOR)
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

    # ── 5. Intent coverage bonus (0.0 – 0.15) ────────────────────────────────
    # Rewards whether the WHOLE episode addressed every required signal at least
    # once. Distinct from step-level rewards. Uses plain bool flags from steps
    # (compatible with both dict and list trajectory formats).
    task_key = task_name.replace("task_easy", "order_status") \
                        .replace("task_medium", "damaged_product") \
                        .replace("task_hard", "escalation")
    required = _TASK_REQUIRED_INTENTS.get(task_key, [])
    if required and steps:
        covered = 0
        for ri in required:
            for s in steps:
                intents_in_step = s.get("detected_intents", {})
                raw = intents_in_step.get(ri)
                if raw is None:
                    # Legacy bool flags directly on the step dict
                    if s.get(ri, False):
                        covered += 1
                        break
                elif isinstance(raw, dict):
                    if raw.get("present", False) and raw.get("confidence", 0.0) >= 0.5:
                        covered += 1
                        break
                elif raw:  # plain bool
                    covered += 1
                    break
        intent_bonus = 0.15 * (covered / len(required))
    else:
        intent_bonus = 0.0

    # ── Total (max = 0.45 + 0.15 + 0.10 + 0.15 + 0.15 = 1.00) ──────────────
    total = resolution_score + efficiency_score + mood_score + quality_score + intent_bonus

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
      +0.20  expected / estimated delivery date mentioned (specific phrases only)
      +0.15  professional greeting or closure phrase
      +0.10  case / reference number given
    Total max = 1.00  (clamped to [0.01, 0.99])

    Note: Generic month/auxiliary words excluded to prevent false positives.
    e.g. "may" (auxiliary verb) and standalone "days" removed — use specific
    date phrases like "estimated delivery", "deliver by", "arrival date".
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
    # Delivery date: require specific date-intent phrases to avoid false positives.
    # "may" (auxiliary verb) and bare "days" removed — they trigger on unrelated text.
    if any(w in text for w in ["expected delivery", "estimated delivery", "delivery date",
                                "arrival date", "arrive by", "deliver by", "arrives on",
                                "business days", "april", "march", "june", "july",
                                "august", "september", "october", "november", "december",
                                "january", "february"]):
        score += 0.20
    if any(w in text for w in ["hello", "hi", "thank you for reaching", "happy to help",
                                "glad to help", "anything else", "have a great",
                                "you're welcome", "my pleasure"]):
        score += 0.15
    if any(w in text for w in ["case number", "reference number", "ticket number",
                                "case id", "ref #", "ref no"]):
        score += 0.10

    return round(max(0.01, min(0.99, score)), 2)


def grade_damaged_product(response: str, state: dict = None) -> float:
    """
    Grade a single agent response for the damaged_product task.

    Scoring (no free base — blank responses score 0.01):
      +0.25  apology / regret expressed
      +0.20  empathy / understanding shown
      +0.25  replacement or refund clearly offered
      +0.15  timeline / ETA given for resolution (specific time phrases only)
      +0.15  case / reference number provided (specific phrases only)
    Total max = 1.00  (clamped to [0.01, 0.99])

    Note: "ref" alone removed — too broad (matches "preferred", "referral")
    Use "reference number", "ref #", "ref no" for case reference credit.
    Bare "days" / "24" / "48" retained for timeline but standalone "week"
    firmed up to "within a week" / "within the week" to reduce false positives.
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
    # Timeline: require specific time-intent phrases; bare "week" removed to avoid
    # false positives like "last week" or "next week" in unrelated context.
    if any(w in text for w in ["business days", "within 24", "within 48", "3-5 days",
                                "within the week", "within a week", "within 3",
                                "within 5", "hours", "timeline", "within",
                                "24 hours", "48 hours"]):
        score += 0.15
    # Case reference: "ref" alone triggers on "preferred", "referral", etc.
    # Require explicit reference-number phrases only.
    if any(w in text for w in ["case number", "reference number", "ticket number",
                                "case id", "ref #", "ref no", "confirmation number",
                                "confirmation code"]):
        score += 0.15

    return round(max(0.01, min(0.99, score)), 2)


def grade_escalation(response: str, state: dict = None) -> float:
    """
    Grade a single agent response for the escalation (hard) task.

    This grader is intentionally demanding — frontier models must work across
    multiple turns to achieve a high score:

      +0.20  genuine apology and empathy demonstrated
      +0.20  refund or compensation explicitly committed
      +0.20  manager / supervisor escalation offered
             (full credit only at step >= 2; partial 0.10 at step 1)
      +0.15  concrete resolution timeline stated
      +0.15  case / reference number provided
      +0.10  strong de-escalation language beyond generic apology
    Total max = 1.00  (clamped to [0.01, 0.99])

    Single-turn cap: If step == 1 (first agent turn), total score is capped at
    0.60 to ensure the hard task cannot be fully solved in one response.
    This forces multi-turn de-escalation → resolution → closure progression.

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

    # 4. Timeline / commitment (specific time-intent phrases)
    if any(w in text for w in ["processed", "within", "business days",
                                "within 48", "within 24", "3-5 days",
                                "by tomorrow", "immediately", "24 hours", "48 hours"]):
        score += 0.15

    # 5. Case / reference number (shows procedural closure)
    if any(w in text for w in ["case number", "reference number", "ticket number",
                                "case id", "confirmation number", "ref #", "ref no"]):
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

    # ── Single-turn hard cap (step 1 only) ───────────────────────────────────
    # Prevents a single perfectly-crafted response from fully solving the hard
    # task. Frontier models must demonstrate multi-turn de-escalation ability.
    if step <= 1:
        score = min(score, 0.60)

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

