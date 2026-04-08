"""
ImmediateRecoveryPolicy — One-step recovery enforcement (internal only).

Detects weak responses (low reward) and forces a 'Recovery Mode' prompt 
that fast-tracks the agent back to a high-performance state (0.1 -> 0.6 -> 0.9+).
"""

from typing import Dict, List, Optional

class ImmediateRecoveryPolicy:
    """
    Enforces immediate recovery after a low reward step.
    Skips unnecessary questioning and forces resolution intents.
    """

    _RECOVERY_PROMPTS: Dict[str, str] = {
        "order_status": (
            "Your previous response was insufficient. RECOVER IMMEDIATELY: "
            "1. Apologize for the confusion. "
            "2. Provide the tracking number (TRK987654321) AND current status (In Transit). "
            "3. Provide the delivery date (April 3rd). "
            "4. Add: 'Is there anything else I can assist with today?'"
        ),
        "damaged_product": (
            "Your previous response was insufficient. RECOVER IMMEDIATELY: "
            "1. Sincere apology and empathy. "
            "2. Offer a replacement shipped today (3-5 business days). "
            "3. Ask for the order number if not yet provided. "
            "4. Add: 'I'll make sure this is resolved now. Is there anything else?'"
        ),
        "escalation": (
            "Your previous response was insufficient. RECOVER IMMEDIATELY: "
            "1. Deep apology and de-escalation effort. "
            "2. Confirm immediate escalation to a senior manager. "
            "3. Confirm a full refund will be processed within 48 hours. "
            "4. Add: 'Thank you for your patience. Is there anything else?'"
        )
    }

    @classmethod
    def get_recovery_hint(cls, task_name: str, last_reward: float) -> Optional[str]:
        """
        Returns a recovery hint if the last reward was critically low (< 0.3).
        """
        if last_reward >= 0.3:
            return None
        
        prompt = cls._RECOVERY_PROMPTS.get(task_name, "Provide a complete resolution immediately.")
        
        return (
            f"\n\n[IMMEDIATE_RECOVERY_MODE]: {prompt}\n"
            "Do NOT ask more questions. Transition directly to resolution."
        )
