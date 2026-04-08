"""
ResponseValidator — Pre-response decision layer (internal only).

Detects and flags 4 failure modes BEFORE the agent response is submitted
to the environment:
  1. repetition   — Jaccard similarity > 0.9 vs last 3 responses
  2. stalling     — consecutive information_request intents > 2 without progress
  3. missing_intent — required intent for current stage absent
  4. off_topic    — exact phrase match for redirect-to-website type deflections

Returns a ValidationResult with correction hints.
This module NEVER appears in API output or the Observation schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set


# ---------------------------------------------------------------------------
# Strict off-topic phrases (same as intents.py off_topic_strict)
# ---------------------------------------------------------------------------
_OFF_TOPIC_PHRASES = [
    "visit our website",
    "check website",
    "go to website",
    "check our website",
    "online portal",
    "please visit",
    "i cannot help",
    "i am unable to help",
    "i'm unable to help",
]

# ---------------------------------------------------------------------------
# Stage required intents (by task × stage)
# Used to detect missing required intents before submission.
# ---------------------------------------------------------------------------
_STAGE_REQUIRED_INTENTS: dict = {
    "order_status": {
        "inquiry":    {"information_provide"},
        "resolution": {"information_provide", "confirmation"},
        "closure":    {"confirmation"},
    },
    "damaged_product": {
        "start":      {"apology"},
        "empathy":    {"de_escalation"},
        "diagnosis":  {"information_request"},
        "resolution": {"resolution_offer"},
        "closure":    {"confirmation"},
    },
    "escalation": {
        "start":          {"apology", "de_escalation"},
        "de_escalation":  {"de_escalation"},
        "acknowledgement":{"resolution_offer"},
        "resolution":     {"resolution_offer", "confirmation"},
        "closure":        {"confirmation"},
    },
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    is_valid: bool
    issue_type: str          # "" | "repetition" | "stalling" | "missing_intent" | "off_topic"
    correction_hint: str     # Human-readable suggestion fed back into agent prompt
    severity: str            # "low" | "medium" | "high"


@dataclass
class ResponseValidatorState:
    """Per-episode state for the validator. Lives in inference.py, not in env."""
    last_responses: List[str] = field(default_factory=list)
    consecutive_info_requests: int = 0
    last_intents: List[Set[str]] = field(default_factory=list)
    last_reward: float = 1.0


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

class ResponseValidator:
    """
    Stateless validator (state passed in via ResponseValidatorState).

    Usage (in inference pipeline):
        state = ResponseValidatorState()
        result = ResponseValidator.validate(
            draft_response, intents, task_name, stage_name, state
        )
        if not result.is_valid:
            # Re-draft or augment based on result.correction_hint
        state = ResponseValidator.update_state(state, draft_response, intents, reward)
    """

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @staticmethod
    def _is_off_topic(text: str) -> bool:
        lower = text.lower()
        return any(p in lower for p in _OFF_TOPIC_PHRASES)

    @classmethod
    def validate(
        cls,
        draft_response: str,
        intents: Set[str],          # bridged intents from get_bridge_intents
        task_name: str,
        stage_name: str,
        state: ResponseValidatorState,
    ) -> ValidationResult:
        """
        Validate a draft response and return a ValidationResult.

        Checks run in priority order: off_topic → repetition → stalling → missing_intent.
        Returns the FIRST problem found (highest-severity first).
        """

        # ── 1. Off-topic ─────────────────────────────────────────────────────
        if cls._is_off_topic(draft_response):
            return ValidationResult(
                is_valid=False,
                issue_type="off_topic",
                correction_hint=(
                    "Your response appears to deflect to external resources. "
                    "Please directly address the customer's concern with a concrete action. "
                    "Do NOT say 'visit our website' or redirect them elsewhere."
                ),
                severity="high",
            )

        # ── 2. Repetition (Jaccard > 0.9 vs last 3) ──────────────────────────
        for prev in state.last_responses[-3:]:
            if cls._jaccard(draft_response, prev) > 0.9:
                return ValidationResult(
                    is_valid=False,
                    issue_type="repetition",
                    correction_hint=(
                        "Your response is too similar to a previous reply. "
                        "Rephrase completely and add new, concrete information. "
                        "Avoid repeating the same phrases or sentences."
                    ),
                    severity="high",
                )

        # ── 3. Stalling (consecutive info_request > 2) ────────────────────────
        if state.consecutive_info_requests > 2 and "information_request" in intents:
            return ValidationResult(
                is_valid=False,
                issue_type="stalling",
                correction_hint=(
                    "You have requested information multiple times without resolution. "
                    "Provide a partial resolution or concrete next step now, "
                    "even if you don't have all the details yet."
                ),
                severity="medium",
            )

        # ── 4. Missing required intent for current stage ───────────────────────
        required = _STAGE_REQUIRED_INTENTS.get(task_name, {}).get(stage_name, set())
        if required and not (intents & required):
            missing = ", ".join(sorted(required - intents))
            return ValidationResult(
                is_valid=False,
                issue_type="missing_intent",
                correction_hint=(
                    f"Your response is missing a required action for the '{stage_name}' "
                    f"stage: {missing}. Make sure your response explicitly includes this."
                ),
                severity="medium",
            )

        return ValidationResult(is_valid=True, issue_type="", correction_hint="", severity="")

    @staticmethod
    def update_state(
        state: ResponseValidatorState,
        response: str,
        intents: Set[str],
        reward: float,
    ) -> ResponseValidatorState:
        """Update per-episode state after a response is submitted."""
        state.last_responses.append(response)
        if len(state.last_responses) > 5:
            state.last_responses = state.last_responses[-5:]

        if "information_request" in intents and "resolution_offer" not in intents:
            state.consecutive_info_requests += 1
        else:
            state.consecutive_info_requests = 0

        state.last_intents.append(set(intents))
        if len(state.last_intents) > 5:
            state.last_intents = state.last_intents[-5:]

        state.last_reward = reward
        return state

    @staticmethod
    def needs_recovery(state: ResponseValidatorState) -> bool:
        """Returns True if the last step reward was critically low (< 0.3)."""
        return state.last_reward < 0.3
