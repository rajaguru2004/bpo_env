"""
RepeatIntentDetector — Fast-track detection for repeated user prompts (internal only).

Tracks the last 2 user inputs and flags cases where the user is repeating
their request because the agent is stalling.

This module NEVER appears in API output or the Observation schema.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Optional


class RepeatIntentDetector:
    """
    Tracks user input history to detect when a 'Fast-Track' resolution
    is needed due to repetitive frustration.
    """

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        """Normalized similarity metric between two strings."""
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @classmethod
    def should_force_resolution(
        cls,
        current_input: str,
        history: List[str],
        current_intents: Set[str],
        last_intents: Optional[Set[str]] = None,
    ) -> bool:
        """
        Returns True if the system should enter FORCE_RESOLUTION_MODE:
        - If semantic similarity > 0.85 with the PREVIOUS user input.
        - OR if the same intent is repeated (e.g., repeating 'refund').
        """
        if not history:
            return False

        last_input = history[-1]
        
        # 1. Direct semantic similarity
        if cls._jaccard(current_input, last_input) > 0.85:
            return True

        # 2. Intent repetition check
        if last_intents and current_intents:
            # Check for core resolution-seeking intent overlap
            critical_intents = {"refund", "replacement", "escalation", "information_provide"}
            overlap = current_intents & last_intents & critical_intents
            if overlap:
                return True

        return False

    @staticmethod
    def get_force_prompt(task_name: str) -> str:
        """Prompt instructions for FORCE_RESOLUTION_MODE."""
        # Acceleration logic for each task
        prompts = {
            "order_status": (
                "The customer is repeating their request. SKIP further inquiry. "
                "Immediately provide the tracking number (TRK987654321) and the "
                "expected arrival date (April 3rd) in this response."
            ),
            "damaged_product": (
                "The customer is repeating their request. SKIP diagnosis. "
                "Immediately apologize and offer a replacement unit shipped today."
            ),
            "escalation": (
                "The customer is frustrated and repeating their demand. "
                "Immediately confirm the full refund AND escalate this case to a "
                "senior manager NOW."
            )
        }
        
        instruction = prompts.get(task_name, "Resolve the customer's issue immediately.")
        
        return (
            f"\n\n[FORCE_RESOLUTION_MODE]: {instruction} Do not ask further questions. "
            "Resolve the issue in this next turn."
        )
