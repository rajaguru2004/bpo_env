"""
ClosureEnforcer — Auto-closure enforcement (internal only).

Ensures that every successful resolution is followed by a clean closure
with the required intents and phrases.
"""

from typing import Dict, List, Optional, Set

class ClosureEnforcer:
    """
    Enforces clean closure after a resolution has been provided.
    """

    _CLOSURE_PHRASES: List[str] = [
        "Is there anything else I can assist with today?",
        "Please let me know if you have any other questions. Have a great day!",
        "I'm here to help if you need further assistance. Thank you for your patience.",
        "Your satisfaction is our priority. Is there anything else I can do for you?"
    ]

    @classmethod
    def get_closure_hint(cls, stage_name: str, current_intents: Set[str]) -> Optional[str]:
        """
        Returns a closure hint if the stage is resolution or if a resolution intent was detected.
        """
        resolution_intents = {"tracking_info", "delivery_info", "refund", "replacement", "escalation"}
        
        has_resolution = any(intent in resolution_intents for intent in current_intents)
        
        if stage_name == "resolution" or has_resolution:
            # Check if closure intent (confirmation) is already present
            if "closure" not in current_intents:
                return (
                    "\n\n[CLOSURE_ENFORCEMENT]: You have provided a resolution. "
                    "You MUST now transition to closure. "
                    "Include a professional closing check: 'Is there anything else I can assist with today?' "
                    "and provide a reference number if applicable."
                )
        
        return None

    @classmethod
    def ensure_closure_phrase(cls, response: str, intents: Set[str]) -> str:
        """
        Internal post-processor concept: if the response lacks closure but has resolution, 
        we could append it. However, per instructions, we prefer doing this via hints 
        first to let the LLM generate it naturally.
        """
        if "closure" not in intents and any(i in {"refund", "replacement", "tracking_info"} for i in intents):
            return response + " " + cls._CLOSURE_PHRASES[0]
        return response
