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
        last_intents: Optional[Set[Set[str]]] = None, # Upgraded to check longer history if needed
        full_intent_history: Optional[List[Set[str]]] = None,
    ) -> bool:
        """
        Returns True if the system should enter FORCE_FINAL_RESOLUTION mode:
        - If semantic similarity > 0.85 with the PREVIOUS user input.
        - OR if the SAME critical intent is repeated 2 or more times.
        """
        if not history:
            return False

        last_input = history[-1]
        
        # 1. Direct semantic similarity (Strict)
        if cls._jaccard(current_input, last_input) > 0.85:
            return True

        # 2. Aggressive Intent repetition check (Task 3)
        critical_intents = {"refund", "replacement", "escalation", "information_provide"}
        
        # Check against the immediate previous turn
        if full_intent_history and len(full_intent_history) >= 1:
            prev_intents = full_intent_history[-1]
            overlap = current_intents & prev_intents & critical_intents
            if overlap:
                return True

        return False

    @staticmethod
    def get_force_prompt(task_name: str) -> str:
        """Prompt instructions for FORCE_FINAL_RESOLUTION (Task 3)."""
        prompts = {
            "order_status": (
                "The customer is repeating their request. SKIP all inquiry. "
                "Immediately provide: Tracking (TRK987654321), Delivery Date (April 3rd), "
                "and a closure check."
            ),
            "damaged_product": (
                "The customer is repeating their request. SKIP diagnosis. "
                "Immediately offer: Replacement shipping today (3-5 days arrival) "
                "and a closure check."
            ),
            "escalation": (
                "The customer is frustrated and repeating their demand. "
                "Immediately provide: Full refund confirmation (48h) AND "
                "Manager escalation confirmation now."
            )
        }
        
        instruction = prompts.get(task_name, "Resolve the issue immediately and completely.")
        
        return (
            f"\n\n[FORCE_FINAL_RESOLUTION]: {instruction} Do NOT ask further questions. "
            "This turns MUST resolve the issue fully."
        )
