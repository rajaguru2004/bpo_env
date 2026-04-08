"""
StageSequenceGuard — Strict intent ordering and sequence enforcement (internal only).

Ensures that the agent follows exact SOP rules (e.g., apology BEFORE diagnosis).
Provides logic to fix or warn about sequence violations in real-time.

This module NEVER appears in API output or the Observation schema.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set


@dataclass
class GuardResult:
    is_ordered: bool
    missing_intent: str
    repair_hint: str


class StageSequenceGuard:
    """
    Validates that necessary intents are present in the correct sequence
    based on the current task and progress.
    """

    _SEQUENCES = {
        "order_status":      ["apology", "information_provide", "confirmation"],
        "damaged_product":   ["apology", "information_request", "resolution_offer", "confirmation"],
        "escalation":        ["apology", "de_escalation", "resolution_offer", "confirmation"],
    }

    @classmethod
    def check_sequence(
        cls,
        task_name: str,
        current_stage: str,
        detected_intents: Set[str],
        history_intents: List[Set[str]],
    ) -> GuardResult:
        """
        Check if the current response (detected_intents) is skipping
        any mandatory preceding steps in the SOP sequence.
        """
        full_seq = cls._SEQUENCES.get(task_name, [])
        if not full_seq:
            return GuardResult(True, "", "")

        # All intents seen so far in the episode
        seen_so_far = set().union(*history_intents) if history_intents else set()
        
        # Combined view of history + current turn
        all_intents = seen_so_far | detected_intents

        # 1. Identify where we SHOULD be in the sequence
        for intent in full_seq:
            if intent not in all_intents:
                # This intent was skipped or not yet reached
                # If we're already trying to do the NEXT intent, we have a violation
                next_intents = full_seq[full_seq.index(intent)+1:]
                if any(ni in detected_intents for ni in next_intents):
                    # Sequence violation — trying to jump ahead!
                    return GuardResult(
                        is_ordered=False,
                        missing_intent=intent,
                        repair_hint=f"Wait! Before you resolve, you MUST include a(n) '{intent}' action."
                    )
                break

        return GuardResult(True, "", "")

    @staticmethod
    def get_repair_injection(missing_intent: str) -> str:
        """Injection strings to auto-repair a response's intent sequence."""
        # Phrases to help manually repair a response without LLM re-drafting
        repairs = {
            "apology":             "I sincerely apologize for the inconvenience this has caused you.",
            "de_escalation":       "I completely understand your frustration and we take this seriously.",
            "information_request": "Could you please confirm your order number?",
            "resolution_offer":    "I will process your refund immediately.",
            "confirmation":        "Is there anything else I can assist with today?"
        }
        return repairs.get(missing_intent, "Please address the customer's previous point.")
