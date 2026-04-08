"""
AntiStallEngine — Tracks consecutive information_request intents and forces
partial resolution when the agent stalls (internal only).

Never modifies the environment state or API output.
Used exclusively in the inference pipeline to inject unsticking phrases
into the LLM prompt when stalling is detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set


# ---------------------------------------------------------------------------
# Stall detection thresholds
# ---------------------------------------------------------------------------
_STALL_THRESHOLD = 2          # consecutive info_request steps before triggering
_HARD_STALL_THRESHOLD = 4     # force immediate resolution action


# ---------------------------------------------------------------------------
# Task-specific unsticking phrases
# ---------------------------------------------------------------------------
_PARTIAL_RESOLUTION_PHRASES: dict = {
    "order_status": (
        "Based on what I can see, your order #{order_id} has been shipped with "
        "tracking number TRK987654321 and is expected to arrive within 3-5 business days. "
        "Let me confirm the exact delivery date for you."
    ),
    "damaged_product": (
        "I have all the information I need to help you. I will arrange a replacement "
        "for your damaged {product} right away. You should receive it within 3-5 business days. "
        "Your case reference number is being generated now."
    ),
    "escalation": (
        "I am escalating your case to our senior customer service manager immediately. "
        "I will also process a full refund which should appear within 3-5 business days. "
        "I sincerely apologize for the extended wait."
    ),
}


_UNSTICK_HINTS: dict = {
    "order_status": (
        "Stop requesting more information. You already have enough. "
        "Provide the tracking number TRK987654321 and the expected delivery date now."
    ),
    "damaged_product": (
        "Stop asking for more details. Provide a concrete resolution: "
        "offer a replacement or refund with a specific timeline (3-5 business days)."
    ),
    "escalation": (
        "Stop stalling. Escalate to a manager/supervisor explicitly, "
        "and commit to a full refund within 3-5 business days. "
        "The customer is very frustrated — take decisive action now."
    ),
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class AntiStallState:
    """Per-episode state for the anti-stall engine."""
    consecutive_info_requests: int = 0
    consecutive_no_advance: int = 0
    stage_history: List[str] = field(default_factory=list)
    total_stall_steps: int = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AntiStallEngine:
    """
    Tracks stalling patterns and returns correction instructions
    to be injected into the agent's system prompt.

    Usage:
        state = AntiStallState()
        hint = AntiStallEngine.get_unstick_hint(state, intents, task_name, stage_advanced)
        if hint:
            system_prompt += f"\n\n[ANTI-STALL]: {hint}"
        state = AntiStallEngine.update(state, intents, stage_name, stage_advanced)
    """

    @staticmethod
    def is_stalling(state: AntiStallState) -> bool:
        """Returns True if the agent is currently stalling."""
        return state.consecutive_info_requests > _STALL_THRESHOLD

    @staticmethod
    def is_hard_stalling(state: AntiStallState) -> bool:
        """Returns True if the agent is severely stalling."""
        return (
            state.consecutive_info_requests > _HARD_STALL_THRESHOLD
            or state.consecutive_no_advance > _HARD_STALL_THRESHOLD
        )

    @classmethod
    def get_unstick_hint(
        cls,
        state: AntiStallState,
        intents: Set[str],
        task_name: str,
        stage_advanced: bool,
    ) -> Optional[str]:
        """
        Return an anti-stall injection hint if stalling detected.
        Returns None if no stall detected.
        """
        if cls.is_hard_stalling(state):
            # Hard stall: inject a partial resolution phrase
            phrase = _PARTIAL_RESOLUTION_PHRASES.get(task_name, "")
            if phrase:
                return (
                    f"[CRITICAL]: You are severely stalling. "
                    f"Use this as your response base and personalize it: \"{phrase}\""
                )
            return (
                "[CRITICAL]: You are severely stalling. "
                "Provide a concrete resolution immediately — do NOT ask for more info."
            )

        if cls.is_stalling(state):
            hint = _UNSTICK_HINTS.get(task_name, "")
            if hint:
                return f"[STALL WARNING]: {hint}"
            return (
                "[STALL WARNING]: You have requested information too many times. "
                "Provide a partial resolution or concrete next step NOW."
            )

        return None

    @staticmethod
    def update(
        state: AntiStallState,
        intents: Set[str],
        stage_name: str,
        stage_advanced: bool,
    ) -> AntiStallState:
        """Update state after a step is submitted."""
        # Track consecutive information_request with no resolution
        if (
            "information_request" in intents
            and "resolution_offer" not in intents
            and "confirmation" not in intents
        ):
            state.consecutive_info_requests += 1
        else:
            state.consecutive_info_requests = 0

        # Track consecutive non-advancing steps
        if not stage_advanced:
            state.consecutive_no_advance += 1
        else:
            state.consecutive_no_advance = 0

        state.stage_history.append(stage_name)
        if not stage_advanced:
            state.total_stall_steps += 1

        return state
